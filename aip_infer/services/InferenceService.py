import os
import glob
import json
import shutil
import base64
import mimetypes
import pandas as pd
from io import StringIO
from zipfile import ZipFile
from django.conf import settings
from django.http import FileResponse
from django.core.files.storage import default_storage
from rest_framework.response import Response
from mlcore.aip_train.kubeflow_manager import KubeflowManager
from mlcore.aip_datapre.pachyderm_repository_manager import PachydermRepositoryManager

from aip import consts
from aip_train.consts import MILLI
from aip.tasks import status_manage
from aip import settings as aip_settings
from aip.daos.preset_dao import PresetDao
from aip.daos.dataset_dao import DatasetDao
from aip.exceptions import DiffFromTrainStats
from aip.daos.algorithm_dao import AlgorithmDao
from aip.daos.inference_dao import InferenceDao
from aip.daos.basic_info_dao import BasicInfoDao
from aip.daos.experiment_dao import ExperimentDao
from aip.daos.docker_image_dao import DockerImageDao
from aip.daos.inference_log_dao import InferenceLogDao
from aip.daos.resource_info_dao import ResourceInfoDao
from aip_train.services.train_service import TrainService
from aip.daos.resource_status_dao import ResourceStatusDao
from aip.utils import FileUtils, FileConvert, DatetimeUtils
from aip_data.services.dataset_service import DatasetService
from aip.daos.tabular_dataset_stats_dao import TabularDatasetStatsDao
from aip.exceptions import InvalidSaveFormatException, InferenceResultNotExists

dataset_service = DatasetService()
train_service = TrainService()


class InferenceService:
    def upload_inference_file(self, inference_id, file_list):
        path = os.path.join("inference", str(inference_id))
        FileUtils.save_file_list(path, file_list)
        _, file_ext = os.path.splitext(file_list[0].name)
        if file_ext in [".jpg", ".png"]:
            InferenceDao.objects.update_file_upload(inference_id, file_format="TYI")
        elif file_ext in [".csv"]:
            InferenceDao.objects.update_file_upload(inference_id, file_format="TYS")

    def get_image_list(self, dataset_id):
        dataset_info = DatasetDao.objects.get_dataset_detail(dataset_id)
        pachyderm_model = PachydermRepositoryManager(repo_name=dataset_info.repo_nm, branch_name=dataset_info.branch_nm)
        image_list = pachyderm_model.list_file("/*")
        return [os.path.basename(chunk.file.path)
                for chunk in image_list if os.path.splitext(chunk.file.path)[1] in ['.jpg', '.png']]

    def get_dataset_user(self, dataset_id):
        dataset_user = DatasetDao.objects.get_dataset_user_id(dataset_id)
        return dataset_user

    def _get_tabular_source_from_pachyderm(self, dataset_id):
        dataset_info = DatasetDao.objects.get_dataset_detail(dataset_id)
        pachyderm_model = PachydermRepositoryManager(repo_name=dataset_info.repo_nm, branch_name=dataset_info.branch_nm)
        source_file_list = pachyderm_model.get_file("/tabular/*")
        return "\n".join([chunk.decode("utf-8") for chunk in source_file_list])

    def get_tabular_data_from_storage(self, dataset_id):
        tabular_source = self._get_tabular_source_from_pachyderm(dataset_id)
        chunk_to_json = pd.read_csv(StringIO(tabular_source)).to_json(orient='records', lines=True)
        return FileConvert.tabular_to_json(chunk_to_json)

    def get_tabular_info(self, dataset_id):
        tabular_source = self._get_tabular_source_from_pachyderm(dataset_id)
        return list(pd.read_csv(StringIO(tabular_source)).columns)

    def get_image(self, dataset_id, file_name):
        dataset_info = DatasetDao.objects.get_dataset_detail(dataset_id)
        pachyderm_model = PachydermRepositoryManager(repo_name=dataset_info.repo_nm, branch_name=dataset_info.branch_nm)
        source_file_list = pachyderm_model.get_file("images/"+file_name)
        return [base64.b64encode(chunk) for chunk in source_file_list]

    def get_inference_log_progress(self, inference_id):
        progress = InferenceLogDao.objects.get_list({"inference_id": inference_id}).first()
        data = {
            "total_count": progress.total_infer_data_count,
            "complete_count": progress.total_complete_data_count,
            "inference_time_per_data": f"{round(progress.infer_time_per_data, 3)}s",
            "rest_time": DatetimeUtils.process_time_to_hms((progress.total_infer_data_count - progress.total_complete_data_count)*progress.infer_time_per_data),
            "inference_log": self.get_inference_log(inference_id, progress),
            "inference_metric": progress.inference_metric.replace('\'', '\"') if progress.inference_metric else None
        }
        return data

    def get_inference_log(self, inference_id, progress):
        if FileUtils.is_log_file_exists(inference_id):
            logs = FileUtils.get_log_from_local_file(inference_id)
            inference_log = [{log} for log in logs.split("\n")]
        else:
            logs = [[log] for log in progress.inference_log.split('\n')]
            inference_log = [set(log) for log in logs]
        return inference_log

    def get_inference_pod_status(self, inference_id, user_id):
        inference_working_status = InferenceDao.objects.get_detail(inference_id).inference_working_status
        pod_status = BasicInfoDao.objects.get_name(inference_working_status)
        if pod_status == "succeeded" and FileUtils.is_log_file_exists(inference_id):
            self._update_inference_log_info(inference_id, user_id)
        return pod_status

    def _update_inference_log_info(self, inference_id, user_id):
        inference_log = FileUtils.get_log_from_local_file(inference_id)
        inference_log_info = {
            "inference_log": inference_log,
            "update_user": user_id
        }
        InferenceLogDao.objects.update_inference_log_info(inference_id, inference_log_info)
        FileUtils.delete_log_file(inference_id)

    def get_local_dataset(self, inference_id):
        inference_data_type = InferenceDao.objects.get_detail(inference_id).inference_data_type
        if inference_data_type == consts.IMAGE:
            return FileUtils.get_image_url_from_local(inference_id)
        else:
            local_dataset = FileUtils.get_tabular_from_local(inference_id)
            if not local_dataset:
                return {
                    "column_num": None,
                    "column_name": None,
                    "tabular_list": None
                }
            return {
                "column_num": len(local_dataset[0].keys()),
                "column_name": local_dataset[0].keys(),
                "tabular_list": local_dataset
            }

    def get_result_dataset(self, inference_id):
        inference_data_type = InferenceDao.objects.get_detail(inference_id).inference_data_type
        if inference_data_type == consts.IMAGE:
            result_dataset = FileUtils.get_result_image_url_from_local(inference_id)
            data_list = []
            for result in result_dataset:
                data_list.append(
                    {
                        "file_name": os.path.basename(result),
                        "url": result
                    }
                )
            return data_list
        else:
            result_dataset = FileUtils.get_result_tabular_from_local(inference_id)
            if not result_dataset:
                return {
                    "column_num": None,
                    "column_name": None,
                    "tabular_list": None
                }
            return {
                "column_num": len(result_dataset[0].keys()),
                "column_name": result_dataset[0].keys(),
                "tabular_list": result_dataset
            }

    def get_inference_dataset(self, inference_id, model_list):
        inference_info = InferenceDao.objects.get_detail(inference_id)
        inference_status = BasicInfoDao.objects.get_name(inference_info.inference_working_status)
        model_nm = ExperimentDao.objects.get_detail(inference_info.experiment_id).input_model_nm
        before_local_dataset, before_dataset_nm, result_dataset = None, None, None
        try:
            before_dataset_nm = DatasetDao.objects.get_detail(inference_info.dataset_id).dataset_nm
        except:
            before_local_dataset = self.get_local_dataset(inference_id)

        if inference_info.inference_working_status == consts.INFERENCE_SUCCEEDED:
            result_dataset = self.get_result_dataset(inference_id)

        return {
            "inference_id": inference_id,
            "inference_status": inference_status,
            "model_list": model_list,
            "model_nm": model_nm,
            "before_local_dataset": before_local_dataset,
            "before_storage_dataset_nm": before_dataset_nm,
            "after_dataset": result_dataset
        }

    def get_inference_setting(self, project_id, experiment_id):
        model_list = ExperimentDao.objects.get_input_model_list(project_id)
        if experiment_id:
            try:
                inference_id = InferenceDao.objects.get_inference_id(experiment_id)
                return self.get_inference_dataset(inference_id, model_list)
            except:
                model_nm = ExperimentDao.objects.get_input_model_name(experiment_id)["model_nm"]
                return {
                    "inference_id": None,
                    "inference_status": None,
                    "model_list": model_list,
                    "model_nm": model_nm,
                    "before_local_dataset": [],
                    "before_storage_dataset_nm": None,
                    "after_dataset": []
                }
        else:
            try:
                inference_id = InferenceDao.objects.get_current_inference_id(project_id)
                return self.get_inference_dataset(inference_id, model_list)
            except:
                return {
                    "inference_id": None,
                    "inference_status": None,
                    "model_list": model_list,
                    "model_nm": None,
                    "before_local_dataset": [],
                    "before_storage_dataset_nm": None,
                    "after_dataset": []
                }

    def get_custom_model_list(self, project_id):
        return ExperimentDao.objects.get_custom_model_list(project_id)

    def update_stop_info(self, inference_id, user_id):
        InferenceDao.objects.update_stop_info(inference_id, user_id)

    def check_results_exists(self, inference_id):
        result_img_data = glob.glob(os.path.join(settings.MEDIA_ROOT, "inference", str(inference_id), "results", "*.jpg"))
        result_tabular_data = glob.glob(os.path.join(settings.MEDIA_ROOT, "inference", str(inference_id), "results", "*.csv"))
        if not result_img_data and not result_tabular_data:
            raise InferenceResultNotExists

    def download_result(self, inference_id, save_format):
        self.check_results_exists(inference_id)
        save_file_name = f"inference_{InferenceDao.objects.get_detail(inference_id).inference_model_nm}.zip"
        path = os.path.join(settings.MEDIA_ROOT, "inference", str(inference_id), "results")
        save_format = f".{save_format}"
        if save_format in [".png", ".jpg"]:
            return FileUtils.image_to_zip(path, save_format, save_file_name)
        elif save_format in [".xlsx", ".csv"]:
            return FileUtils.tabular_to_zip(path, save_format, save_file_name)
        else:
            raise InvalidSaveFormatException

    def add_inference_id(self, inference_info):
        InferenceDao.objects.add_inference_info(inference_info)
        return InferenceDao.objects.get_inference_id(inference_info["experiment_id"])

    def add_inference_log_info(self, inference_id, user_id):
        inference_log_info = {
            "inference_id": inference_id,
            "create_user": user_id
        }
        InferenceLogDao.objects.add_inference_log_info(inference_log_info)

    def update_inference_local_filename(self, inference_id):
        local_path = '\r\n'.join(os.listdir(os.path.join(settings.MEDIA_ROOT, "inference", str(inference_id))))
        InferenceLogDao.objects.update_inference_local_filename(inference_id, local_path)

    def start_inference(self, inference_info, inference_id, annotation_data=None):
        previous_inferences = InferenceDao.objects.get_list({"project_id": inference_info["project_id"]})
        if len(previous_inferences) > 1 and not annotation_data:
            previous_inference_id = previous_inferences[len(previous_inferences)-2].inference_id
            self.delete_local_file(previous_inference_id)
        experiment_info = ExperimentDao.objects.get_detail(inference_info["experiment_id"])
        trained_dataset_info = DatasetDao.objects.get_detail(experiment_info.dataset_id)
        if trained_dataset_info.dataset_type == consts.IMAGE:
            self.start_image_inference(inference_info, inference_id, experiment_info, trained_dataset_info, annotation_data)
        else:
            self.start_tabular_inference(inference_info, inference_id, experiment_info, trained_dataset_info)

    def start_image_inference(self, inference_info, inference_id, experiment_info, trained_dataset_info, annotation_data=None):
        algorithm_nm, problem_type, framework_type = self.get_algorithm_info(experiment_info.algorithm_id)
        dataset_id = inference_info["dataset_id"]
        dockerimage_nm = DockerImageDao.objects.get_docker_image_nm(problem_type, framework_type, consts.INFERENCE_STEP)
        preset_id = PresetDao.objects.get_inference_preset(step_id=consts.INFERENCE_STEP)["preset_id"]

        pipeline_params = {
            "run-uuid": experiment_info.run_uuid,
            "inference-id": inference_id,
            "algorithm-nm": algorithm_nm,
            "local-file-path": aip_settings.SM_AIP_HOST_DOMAIN,
            "problem-type": problem_type,
            "train-pachy-repo-name": trained_dataset_info.repo_nm,
            "train-pachy-branch-name": trained_dataset_info.branch_nm,
        }
        if annotation_data:
            pipeline_params["annotator-start-idx"] = annotation_data["start_idx"]
            pipeline_params["annotator-end-idx"] = annotation_data["end_idx"]
            total_img = annotation_data["end_idx"] - annotation_data["start_idx"] + 1

        if dataset_id:
            dataset_info = DatasetDao.objects.get_detail(dataset_id)
            total_img = dataset_info.data_count
            InferenceDao.objects.update_file_upload(inference_id, file_format=dataset_info.dataset_type)
            pipeline_params["pachy-repo-name"] = dataset_info.repo_nm
            pipeline_params["pachy-branch-name"] = dataset_info.branch_nm
        else:
            total_img = len(os.listdir(os.path.join(settings.MEDIA_ROOT, "inference", str(inference_id))))

        preset = PresetDao.objects.get_preset_summary(preset_id)
        gpu_size = preset["gpu_size"]
        cpu_size = str(preset["cpu_size"] * MILLI) + "m"
        memory_size = str(preset["memory_size"]) + "Mi"
        node_key, node_value = train_service.get_node_info(preset_id)

        kfp_manager = KubeflowManager(kubeflow_server_address=aip_settings.KUBEFLOW_SERVER_ADDRESS)
        pipeline_id = kfp_manager.upload_pipeline(
            dockerimage_nm=dockerimage_nm,
            model_nm=f"inference_{algorithm_nm}",
            params=pipeline_params,
            gpu_size=gpu_size,
            cpu_size=cpu_size,
            memory_size=memory_size,
            node_key=node_key,
            node_value=node_value,
            mlflow_s3_endpoint_url=aip_settings.MLFLOW_S3_ENDPOINT_URL.split("//")[1],
            mlflow_tracking_uri=aip_settings.MLFLOW_TRACKING_URI,
            minio_bucket_name=aip_settings.MINIO_BUCKET_NAME,
            aws_access_key_id=aip_settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=aip_settings.AWS_SECRET_ACCESS_KEY,
            aws_default_region=aip_settings.AWS_DEFAULT_REGION,
            pachyderm_host_uri=aip_settings.PACHYDERM_SERVER_HOST,
            pachyderm_port=aip_settings.PACHYDERM_SERVER_PORT,
            sm_aip_db_name=aip_settings.DATABASES["default"]["NAME"],
            sm_aip_db_user=aip_settings.DATABASES["default"]["USER"],
            sm_aip_db_password=aip_settings.DATABASES["default"]["PASSWORD"],
            sm_aip_db_host=aip_settings.DATABASES["default"]["HOST"],
            sm_aip_db_port=aip_settings.DATABASES["default"]["PORT"]
        )
        experiment_name = "smr"

        run_id = kfp_manager.run_pipeline(
            model_nm=f"inference_{algorithm_nm}",
            pipeline_id=pipeline_id,
            experiment_name=experiment_name,
            experiment_namespace=aip_settings.NAMESPACE)

        update_info = self.set_inference_init_info(dataset_id, run_id, inference_info, experiment_info.model_nm, preset_id)
        update_log_info = self.set_inference_init_log(inference_info["user_id"], total_img)
        InferenceLogDao.objects.update_inference_log_info(inference_id, update_log_info)
        InferenceDao.objects.update_inference_info(inference_id, update_info)
        status_manage.update_inference_status.apply_async(
            args=[inference_id, run_id, inference_info["user_id"]],
            queue="high_priority"
        )

    def set_inference_init_info(self, dataset_id, run_id, inference_info, model_nm, preset_id):
        return {
            "dataset_id": dataset_id,
            "run_id": run_id,
            "experiment_id": inference_info["experiment_id"],
            "inference_model_nm": model_nm,
            "cleaning_complete_yn": False,
            "update_user": inference_info["user_id"],
            "preset_id": preset_id
        }

    def set_inference_init_log(self, user_id, total_count=None, metric_list=None):
        return {
            "update_user": user_id,
            "inference_log": "Container Creating...",
            "total_infer_data_count": total_count,
            "total_complete_data_count": 0,
            "infer_time_per_data": 0,
            "inference_metric": metric_list
        }

    def get_tabular_column_from_storage(self, dataset_id):
        tabular_stat = TabularDatasetStatsDao.objects.get_tabular_dataset_stats(dataset_id)
        column_list = [stats["feature_nm"] for stats in json.loads(tabular_stat.feature_stats)]
        if column_list[0] == "Unnamed: 0":
            return column_list[1:]
        else:
            return column_list

    def check_feature_name(self, train_column, inference_column, train_target_label):
        if train_column == inference_column or train_target_label:
            return consts.LABEL_EXISTS
        elif train_column[:-1] == inference_column:
            return consts.LABEL_NONE
        else:
            raise DiffFromTrainStats

    def get_preprocessed_columns(self, inference_column, method_list):
        for method in method_list:
            for key, value in method.items():
                if key.startswith("Drop Columns"):
                    column_list = [inference_column[index-1] for index in value["feature_indices"]]
                    return list(filter(lambda x: x not in column_list, inference_column))
                else:
                    return inference_column

    def get_algorithm_info(self, algorithm_id: int) -> list:
        _model_info = AlgorithmDao.objects.get_algorithm_info(algorithm_id)
        return [_model_info["algorithm_nm"], _model_info["model"], _model_info["framework"]]

    def start_tabular_inference(self, inference_info, inference_id, experiment_info, trained_dataset_info):
        preset_id = PresetDao.objects.get_inference_preset(step_id=consts.NON_GPU_STEP)["preset_id"]
        algorithm_nm, problem_type, framework_type = self.get_algorithm_info(experiment_info.algorithm_id)
        dataset_id = inference_info["dataset_id"]
        dockerimage_nm = DockerImageDao.objects.get_docker_image_nm(problem_type, framework_type, consts.INFERENCE_STEP)

        trained_dataset_column = self.get_tabular_column_from_storage(experiment_info.dataset_id)
        train_target_label = DatasetDao.objects.get_detail(experiment_info.dataset_id).label_nm
        pipeline_params = {
            "run-uuid": experiment_info.run_uuid,
            "inference-id": inference_id,
            "algorithm-nm": algorithm_nm,
            "local-file-path": aip_settings.SM_AIP_HOST_DOMAIN,
            "train-pachy-repo-name": trained_dataset_info.repo_nm,
            "train-pachy-branch-name": trained_dataset_info.branch_nm,
        }

        method_list = dataset_service.get_template_preprocess_method(experiment_info.dataset_id)
        if dataset_id:
            dataset_info = DatasetDao.objects.get_detail(dataset_id)
            InferenceDao.objects.update_file_upload(inference_id, file_format=dataset_info.dataset_type)
            pipeline_params["pachy-repo-name"] = dataset_info.repo_nm
            pipeline_params["pachy-branch-name"] = dataset_info.branch_nm
            infer_method_list = dataset_service.get_template_preprocess_method(dataset_id)
            method_list = list(filter(lambda x: x not in infer_method_list, method_list))
            dataset_column = self.get_tabular_column_from_storage(dataset_id)
        else:
            dataset_column = FileUtils.get_tabular_column_from_local(inference_id)

        if method_list:
            dataset_column = self.get_preprocessed_columns(dataset_column, method_list)
        pipeline_params["label-yn"] = self.check_feature_name(trained_dataset_column, dataset_column, train_target_label)

        update_log_info = self.set_inference_init_log(inference_info["user_id"], 0, method_list)
        InferenceLogDao.objects.update_inference_log_info(inference_id, update_log_info)

        preset = PresetDao.objects.get_preset_summary(preset_id)
        gpu_size = preset["gpu_size"]
        cpu_size = str(preset["cpu_size"] * MILLI) + "m"
        memory_size = str(preset["memory_size"]) + "Mi"
        node_key, node_value = train_service.get_node_info(preset_id)

        kfp_manager = KubeflowManager(kubeflow_server_address=aip_settings.KUBEFLOW_SERVER_ADDRESS)
        pipeline_id = kfp_manager.upload_cpu_only_pipeline(
            dockerimage_nm=dockerimage_nm,
            model_nm=f"inference_{algorithm_nm}",
            params=pipeline_params,
            cpu_size=cpu_size,
            memory_size=memory_size,
            node_key=node_key,
            node_value=node_value,
            mlflow_s3_endpoint_url=aip_settings.MLFLOW_S3_ENDPOINT_URL.split("//")[1],
            mlflow_tracking_uri=aip_settings.MLFLOW_TRACKING_URI,
            minio_bucket_name=aip_settings.MINIO_BUCKET_NAME,
            aws_access_key_id=aip_settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=aip_settings.AWS_SECRET_ACCESS_KEY,
            aws_default_region=aip_settings.AWS_DEFAULT_REGION,
            pachyderm_host_uri=aip_settings.PACHYDERM_SERVER_HOST,
            pachyderm_port=aip_settings.PACHYDERM_SERVER_PORT,
            sm_aip_db_name=aip_settings.DATABASES["default"]["NAME"],
            sm_aip_db_user=aip_settings.DATABASES["default"]["USER"],
            sm_aip_db_password=aip_settings.DATABASES["default"]["PASSWORD"],
            sm_aip_db_host=aip_settings.DATABASES["default"]["HOST"],
            sm_aip_db_port=aip_settings.DATABASES["default"]["PORT"]
        )
        experiment_name = "smr"

        run_id = kfp_manager.run_pipeline(
            model_nm=f"inference_{algorithm_nm}",
            pipeline_id=pipeline_id,
            experiment_name=experiment_name,
            experiment_namespace=aip_settings.NAMESPACE)

        update_info = self.set_inference_init_info(dataset_id, run_id, inference_info, experiment_info.model_nm, preset_id)
        InferenceDao.objects.update_inference_info(inference_id, update_info)
        status_manage.update_inference_status.apply_async(
            args=[inference_id, run_id, inference_info["user_id"]],
            queue="high_priority"
        )

    def is_inference_running(self, project_id):
        running_info = InferenceDao.objects.is_inference_running(project_id)
        running_experiment = running_info.experiment_id if running_info else running_info
        return {
            "other_running": bool(running_info),
            "experiment_id": running_experiment
        }

    def serve_result_file(self, file_path):
        if file_path.endswith(".csv"):
            chunk_to_json = pd.read_csv(file_path).to_json(orient="records", lines=True)
            data = {
                "csv": FileConvert.tabular_to_json(chunk_to_json)
            }
            return Response(data)
        return FileResponse(open(file_path, "rb"), content_type=mimetypes.guess_type(file_path)[0])

    def upload_result_file(self, file_path, files):
        result_dir = os.path.join(file_path, "results")
        os.makedirs(result_dir, exist_ok=True)
        if files[0].name.startswith("log"):
            FileUtils.save_file_list(result_dir, files)
        else:
            default_storage.save(os.path.join(result_dir, files[0].name), files[0])
            self.unzip(result_dir, "results.zip")

    def delete_local_file(self, previous_inference_id):
        previous_inference_dir = os.path.join(settings.MEDIA_ROOT, "inference", str(previous_inference_id))
        shutil.rmtree(previous_inference_dir, ignore_errors=True)

    def unzip(self, path, file_name):
        with ZipFile(os.path.join(path, file_name), "r") as zip_file:
            zip_file.extractall(path)
            os.remove(os.path.join(path, file_name))
