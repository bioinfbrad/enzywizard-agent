"""支持通过 ``python -m enzywizard_agent`` 启动命令行程序。"""

from .cli import main

# 只有直接以模块方式运行时才进入 CLI；被其他模块导入时不会自动执行。
if __name__ == "__main__":
    main()
