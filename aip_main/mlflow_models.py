from django.db import models


class Experiments(models.Model):
    experiment_id = models.AutoField(primary_key=True)
    name = models.CharField(unique=True, max_length=256)
    artifact_location = models.CharField(max_length=256, blank=True, null=True)
    lifecycle_stage = models.CharField(max_length=32, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'experiments'

class Runs(models.Model):
    run_uuid = models.CharField(primary_key=True, max_length=32)
    name = models.CharField(max_length=250, blank=True, null=True)
    source_type = models.CharField(max_length=20, blank=True, null=True)
    source_name = models.CharField(max_length=500, blank=True, null=True)
    entry_point_name = models.CharField(max_length=50, blank=True, null=True)
    user_id = models.CharField(max_length=256, blank=True, null=True)
    status = models.CharField(max_length=9, blank=True, null=True)
    start_time = models.BigIntegerField(blank=True, null=True)
    end_time = models.BigIntegerField(blank=True, null=True)
    source_version = models.CharField(max_length=50, blank=True, null=True)
    lifecycle_stage = models.CharField(max_length=20, blank=True, null=True)
    artifact_uri = models.CharField(max_length=200, blank=True, null=True)
    experiment = models.ForeignKey(Experiments, models.DO_NOTHING, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'runs'


class Tags(models.Model):
    key = models.CharField(primary_key=True, max_length=250)
    value = models.CharField(max_length=5000, blank=True, null=True)
    run_uuid = models.ForeignKey(Runs, models.DO_NOTHING, db_column='run_uuid')

    class Meta:
        abstract = True
        managed = False
        db_table = 'tags'
        unique_together = (('key', 'run_uuid'),)

