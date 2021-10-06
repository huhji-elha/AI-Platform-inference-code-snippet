import time
import datetime
from django.db import transaction
from mlcore.aip_train.kubeflow_manager import KubeflowManager

from aip import consts
from aip.daos.alarm_dao import AlarmDao
from aip.daos.inference_dao import InferenceDao
from aip.settings import KUBEFLOW_SERVER_ADDRESS
from aip.daos.inference_log_dao import InferenceLogDao


class StatusService:
    def __init__(self, inference_id, run_id, user_id):
        self.inference_id = inference_id
        self.run_id = run_id
        self.user_id = user_id
        self.kfp_manager = KubeflowManager(kubeflow_server_address=KUBEFLOW_SERVER_ADDRESS)
        self.time_out = time.time() + consts.INFERENCE_STATUS_CHECK_TIMEOUT_IN_SEC
        self.now = None

    def check_kubeflow_pod_status(self):
        while time.time() < self.time_out:
            try:
                self.now = datetime.datetime.now()
                pod_info = self.get_pod_info()
                if pod_info["pod_phase"] == consts.INFERENCE_POD_SUCCEEDED:
                    self.update_pod_succeeded()
                    break
                elif pod_info["pod_phase"] == consts.INFERENCE_POD_PENDING:
                    self.update_pod_pending()
                elif pod_info["pod_phase"] == consts.INFERENCE_POD_FAILED or \
                    consts.INFERENCE_POD_MESSAGE_IMAGEPULLBACKOFF in pod_info["pod_message"] or \
                    consts.INFERENCE_POD_MESSAGE_IMAGEINSPECTERROR in pod_info["pod_message"] or \
                    consts.INFERENCE_POD_MESSAGE_INVALIDIMAGENAME in pod_info["pod_message"]:
                    self.update_pod_failed(pod_info["pod_log"])
                    break
                time.sleep(consts.INFERENCE_WAIT_FOR_STATUS_CHECK_TIME)
            except:
                time.sleep(consts.INFERENCE_WAIT_FOR_STATUS_CHECK_TIME)
                continue

    def get_pod_info(self):
        pod_info = self.kfp_manager.get_pod_info(self.run_id)
        return {
            "now": self.now,
            "pod_log": self.kfp_manager.get_train_log(self.run_id),
            "pod_message": pod_info["pod_message"],
            "user_id": self.user_id,
            "run_id": self.run_id,
            "inference_id": self.inference_id,
            "pod_phase": pod_info["pod_phase"]
        }

    def update_pod_succeeded(self):
        InferenceDao.objects.set_item(
            self.inference_id,
            {"inference_working_status": consts.INFERENCE_SUCCEEDED}
        )
        self.kfp_manager.delete_run(self.run_id)

    def update_pod_pending(self):
        with transaction.atomic():
            InferenceDao.objects.set_item(
                self.inference_id,
                {"inference_working_status": consts.INFERENCE_PENDING}
            )
            InferenceLogDao.objects.update_inference_log_info(
                self.inference_id,
                {"inference_log": "Pending..."}
            )

    def update_pod_failed(self, pod_log):
        with transaction.atomic():
            self.add_alram({"log_detail": pod_log})
            self.update_inference_log({"inference_log": pod_log.split("\n")[-2]})
            self.update_inference_working_status({
                "inference_working_status": consts.INFERENCE_FAILED
            })
        self.kfp_manager.delete_run(self.run_id)

    def update_inference_working_status(self, inference_working_status):
        InferenceDao.objects.set_item(
            self.inference_id,
            inference_working_status
        )

    def update_inference_log(self, inference_log):
        InferenceLogDao.objects.update_inference_log_info(
            self.inference_id,
            inference_log
        )

    def add_alram(self, error_log):
        AlarmDao.objects.create_alarm_to_db(
            {
                "error_id": "EIN-000",
                "exe_id": self.inference_id,
                "project_id": InferenceDao.objects.get_detail(self.inference_id).project_id,
                "count": 0,
                "log_detail": error_log,
                "update_date": self.now,
                "create_user": self.user_id,
                "update_user": self.user_id
            }
        )
