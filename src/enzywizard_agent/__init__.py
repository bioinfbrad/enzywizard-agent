"""EnzyWizard-Agent 顶层包，对外只暴露当前软件版本。"""

from .version import __version__

# 使用 __all__ 限制通配符导入时暴露的公共名称。
__all__ = ["__version__"]
