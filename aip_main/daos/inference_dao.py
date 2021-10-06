from aip import consts
from aip.utils import DatetimeUtils
from querybuilder.query import Query
from aip.models import TbInferenceInfo
from .query_set_base import QuerySetBase


class InferenceQuerySet(QuerySetBase):
    def has_running_inference(self, project_id):
        return self.filter(inference_working_status__in=[consts.INFERENCE_PENDING, consts.INFERENCE_RUNNING], project_id=project_id, annotation_yn=consts.NOT_ANNOTATION).exists()

    def add_inference_info(self, inference_info):
        date = DatetimeUtils.get_current_time()
        if inference_info["dataset_id"]:
            self.create(
                user_id=inference_info["user_id"],
                experiment_id=inference_info["experiment_id"],
                project_id=inference_info["project_id"],
                dataset_id=inference_info["dataset_id"],
                start_date=date,
                create_date=date,
                create_user=inference_info["user_id"],
                update_date=date,
                update_user=inference_info["user_id"],
                annotation_yn=inference_info.get('annotation_yn'),
            )
        else:
            self.create(
                user_id=inference_info["user_id"],
                experiment_id=inference_info["experiment_id"],
                project_id=inference_info["project_id"],
                start_date=date,
                create_date=date,
                create_user=inference_info["user_id"],
                update_date=date,
                update_user=inference_info["user_id"],
                annotation_yn=inference_info.get('annotation_yn'),
            )

    def update_inference_info(self, inference_id, update_info):
        options = {
            "dataset_id": update_info["dataset_id"],
            "user_id": update_info["update_user"],
            "experiment_id": update_info["experiment_id"],
            "run_id": update_info["run_id"],
            "inference_model_nm": update_info["inference_model_nm"],
            "inference_working_status": consts.INFERENCE_RUNNING,
            "cleaning_complete_yn": update_info["cleaning_complete_yn"],
            "preset_id": update_info["preset_id"],
            "complete_date": DatetimeUtils.get_current_time(),
            "create_user": update_info["update_user"],
            "update_date": DatetimeUtils.get_current_time(),
            "update_user": update_info["update_user"],
        }
        self.set_item(inference_id, options)

    def update_stop_info(self, inference_id, user_id):
        options = {
            "stop_date": DatetimeUtils.get_current_time(),
            "update_user": user_id,
            "inference_working_status": consts.INFERENCE_FAILED,
        }
        self.set_item(inference_id, options)

    def update_file_upload(self, inference_id, file_format: bool):
        options = {"inference_data_type": file_format, "upload_dataset_yn": True}
        self.set_item(inference_id, options)

    def get_inference_id(self, experiment_id):
        return self.filter(experiment_id=experiment_id).last().inference_id

    def get_current_inference_id(self, project_id):
        return self.filter(project_id=project_id).last().inference_id

    def is_inference_running(self, project_id):
        return self.filter(
            project_id=project_id, inference_working_status=consts.INFERENCE_RUNNING, annotation_yn=consts.NOT_ANNOTATION
        ).last()

    def get_previous_inference_id(self, project_id, user_id):
        return self.filter(project_id=project_id, user_id=user_id).last().inference_id

    def get_last_running_inference(self, project_id):
        return self.filter(inference_working_status__in=[consts.INFERENCE_PENDING, consts.INFERENCE_RUNNING], annotation_yn=consts.NOT_ANNOTATION, project_id=project_id).latest("create_date")

    def get_inference_task(self, inference_id):
        query = Query().from_table(
            table="TB_INFERENCE_INFO",
            fields=['inference_id'],
        ).where(inference_id=inference_id)
        query.join(
            fields=['project_id'],
            right_table="TB_PROJECT_INFO",
            join_type="INNER JOIN",
            condition="TB_INFERENCE_INFO.project_id = TB_PROJECT_INFO.project_id"
        )
        query.join(
            fields=["codelevel2_nm AS task"],
            right_table={"T1": "TB_BASIC_INFO"},
            join_type="LEFT JOIN",
            condition="TB_PROJECT_INFO.model = T1.codelevel2_id"
        )
        return query.select()[0]["task"]


class InferenceDao(TbInferenceInfo):
    objects = InferenceQuerySet.as_manager()
