# -*- coding: utf-8 -*-
import pendulum
from datetime import timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_PY = "/mnt/d/PhanTichDuLieu/.venv/bin/python"  # python của project

SCR_CRAWL = "/mnt/d/PhanTichDuLieu/scripts/crawler.py"
SCR_MERGE = "/mnt/d/PhanTichDuLieu/scripts/merge_incremental.py"
SCR_CLEAN = "/mnt/d/PhanTichDuLieu/scripts/process_data.py"
SCR_TRAIN = "/mnt/d/PhanTichDuLieu/scripts/train_model.py"

tz = pendulum.timezone("Asia/Ho_Chi_Minh")
default_args = {"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="bds_q7_daily",
    default_args=default_args,
    start_date=pendulum.datetime(2025, 8, 29, tz=tz),
    schedule_interval="0 6 * * *",   # 06:00 mỗi ngày
    catchup=False,
    max_active_runs=1,
    tags=["bds","q7","rental"],
) as dag:

    crawl = BashOperator(task_id="crawl_data", bash_command=f"{PROJECT_PY} {SCR_CRAWL}")
    merge = BashOperator(task_id="merge_incremental", bash_command=f"{PROJECT_PY} {SCR_MERGE}")
    clean = BashOperator(task_id="process_clean", bash_command=f"{PROJECT_PY} {SCR_CLEAN}")
    train = BashOperator(task_id="train_model", bash_command=f"{PROJECT_PY} {SCR_TRAIN}")

    crawl >> merge >> clean >> train
