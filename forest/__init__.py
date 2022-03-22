"""Library of simple routines."""

from forest.witchcoven import Witch
from forest.data import Kettle
from forest.victims import Victim

from .options import options


# A: Kettle在data文件夹中，其余文件较易找到
__all__ = ['Victim', 'Witch', 'Kettle', 'options']
