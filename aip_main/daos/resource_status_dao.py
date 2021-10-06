from aip.models import TbResourceStatus
from .query_set_base import QuerySetBase
from django.db.models.query import QuerySet
from aip.exceptions import NodeNotFoundException


class ResourceStatusQueryset(QuerySetBase):
    def get_node_id(self, cpu_size, memory_size, gpu_size):
        resource_stats = (
            TbResourceStatus.objects.filter(
                cpu_able_amount__gte=cpu_size,
                memory_able_amount__gte=memory_size,
                gpu_able_amount__gte=gpu_size,
            )
            .order_by("-gpu_able_amount")
            .first()
        )
        if resource_stats is None:
            raise NodeNotFoundException

        return resource_stats.node_id

    def get_gpu_node_id(self, cpu_size, memory_size, gpu_size, gpu_node_ids):
        resource_stats = (
            TbResourceStatus.objects.filter(
                cpu_able_amount__gte=cpu_size,
                memory_able_amount__gte=memory_size,
                gpu_able_amount__gte=gpu_size,
                node_id__in=gpu_node_ids
            )
            .order_by("-gpu_able_amount")
            .first()
        )
        if resource_stats is None:
            raise NodeNotFoundException

        return resource_stats.node_id
             
class ResourceStatusDao(TbResourceStatus):
    objects = ResourceStatusQueryset.as_manager()
