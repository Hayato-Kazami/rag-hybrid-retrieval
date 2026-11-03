.PHONY: install info demo eval test clean

install:
	pip install -e .

info:
	python -m ragmini.cli info

demo:
	python -m ragmini.cli ask "机器人电量低于多少要强制退出任务队列" --show-context

eval:
	python -m ragmini.cli eval --modes dense,sparse,hybrid,hybrid+rerank

test:
	pytest -q

clean:
	rm -rf .pytest_cache src/*.egg-info
