from aip.utils import DatetimeUtils
from aip.models import TbInferenceLog
from .query_set_base import QuerySetBase


class InferenceLogQuerySet(QuerySetBase):
    def get_inference_log(self, inference_id):
        return self.get(inference_id=inference_id).inference_log

    def get_inference_log_info(self, inference_id):
        return self.get(inference_id=inference_id)

    def add_inference_log_info(self, inference_log_info):
        date = DatetimeUtils.get_current_time()
        self.create(
            inference_id=inference_log_info["inference_id"],
            create_date=date,
            create_user=inference_log_info["create_user"],
            update_user=inference_log_info["create_user"],
        )

    def update_inference_local_filename(self, inference_id, local_data_path):
        date = DatetimeUtils.get_current_time()
        self.filter(inference_id=inference_id).update(
            local_data_path=local_data_path,
            update_date=date
        )

    def update_inference_log_info(self, inference_id, update_log_info):
        update_log_info["update_date"] = DatetimeUtils.get_current_time()
        self.filter(inference_id=inference_id).update(**update_log_info)


class InferenceLogDao(TbInferenceLog):
    objects = InferenceLogQuerySet.as_manager()
