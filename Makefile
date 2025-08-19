init:
	python3 -m pip install --user pipenv
	pipenv --python 3.7
	pipenv install --dev --skip-lock
	echo "Check the printed version for numpy is 1.16.4" && pipenv run pip list | grep "numpy"
	echo "Check the printed version for torch is 1.4.0" && pipenv run pip list | grep "torch"
	echo "Check the printed version for tensorboard is 1.14.0" && pipenv run pip list | grep "tensorboard"

LIB_LINT = ECGDL
RUN_LINT = run_*.py
PYTHON_LINT = $(LIB_LINT) $(RUN_LINT)

lint: linter_version ending flake8 pylint mypy

linter_version:
	pipenv run pip list | grep -P "(flake8|pyflakes|pycodestyle)"
	pipenv run pip list | grep -P "(pylint|astroid)"
	pipenv run pip list | grep -P "(mypy|typing|typed)"

ending:
	! grep -rHnP --include="*.py" --include="*.json" --include="*.md" --include="*.csv" "\x0D" ${PYTHON_LINT}

flake8:
	pipenv run flake8 ${PYTHON_LINT}

pylint:
	pipenv run pylint ${PYTHON_LINT}

clean_mypy:
	rm -rf .mypy_cache/

mypy: clean_mypy
	pipenv run mypy ${PYTHON_LINT}

clean_log:
	rm ./logs/*.log

run_experiment:
	echo "Setting file descriptor soft limit to 102400" && ulimit -n 102400 && ulimit -Sn && pipenv run python3 run_experiment.py

run_dbdemo:
	pipenv run python3 run_dbdemo.py

run_create_db:
	pipenv run python3 run_create_db.py

run_cam:
	pipenv run python3 run_cam.py

run_gen_split_flag:
	pipenv run python3 run_gen_split_flag.py

run_create_mitbih:
	pipenv run python3 run_create_mitbih.py
