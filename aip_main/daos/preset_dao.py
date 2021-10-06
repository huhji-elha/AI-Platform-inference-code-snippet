from aip.consts import TRAIN_CODE
from aip.models import TbPresetInfo
from .query_set_base import QuerySetBase
from aip_train.consts import TABULAR_PRESET


class PresetQuerySet(QuerySetBase):
    def get_preset_summary(self, preset_id):
        return self.values(
            "preset_id", "preset_nm", "gpu_type", "gpu_display", "gpu_size", "cpu_size", "memory_size"
        ).get(preset_id=preset_id)

    def get_annotation_preset(self, step_id):
        return self.filter(step_id=step_id).values("preset_id", "preset_nm", "gpu_type", "gpu_display", "gpu_size", "cpu_size", "memory_size")[0]

    def get_preset_list(self):
        return self.filter(step_id=TRAIN_CODE).order_by("sort_order").values(
            "preset_id",
            "preset_nm",
            "gpu_type",
            "gpu_size",
            "gpu_display",
            "cpu_size",
            "memory_size",
            "hdd_size",
            "sort_order",
        )

    def get_tabular_preset_list(self):
        return self.filter(preset_id=TABULAR_PRESET).values(
            "preset_id",
            "preset_nm",
            "gpu_type",
            "gpu_size",
            "gpu_display",
            "cpu_size",
            "memory_size",
            "hdd_size",
            "sort_order",
        )

    def get_inference_preset(self, step_id):
        return self.filter(step_id=step_id).values("preset_id").last()


class PresetDao(TbPresetInfo):
    objects = PresetQuerySet.as_manager()
