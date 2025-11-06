
import dataclasses
from typing import Dict, Any, Type, TypeVar
import importlib
import json
from pathlib import Path
import yaml

class ConfigError(Exception):
    """配置系统基础异常"""
    pass


class ConfigValidationError(ConfigError):
    """配置验证失败异常"""

    def __init__(self, message: str, field: str = None, value=None):
        super().__init__(message)
        self.field = field
        self.value = value
        self.message = message

    def __str__(self):
        if self.field:
            return f"配置验证失败 - 字段 '{self.field}': {self.message} (值: {self.value})"
        return f"配置验证失败: {self.message}"


def objtostr(obj):
    _class = obj.__class__
    return f"{_class.__module__}:{_class.__name__}"


def dict_to_config(source: Dict[str, Any], target: Any):
    target_fields = {field.name for field in dataclasses.fields(target)}
    for field in target_fields:
        if field in source.keys() and dataclasses.is_dataclass(getattr(target, field)):
            dict_to_config(source[field], getattr(target, field))
        elif field in source.keys():
            setattr(target, field, source[field])
        else:
            # print(f"[WARNING] field {field} not found in source config")
            pass

def instantiate_config(config: str) -> object:
    """Convert an object to a file path based on its module and class name."""
    modules_name, class_name = config._target_.split(":")
    modules = importlib.import_module(modules_name)
    assert hasattr(modules, class_name), f"Module {modules_name} has no attribute {class_name}"

    class_obj = getattr(modules, class_name)
    return class_obj()


T = TypeVar('T', bound='BaseConfig')

@dataclasses.dataclass
class BaseConfig:
    """MLP前向表示网络配置类"""
    _config_class_name_: str = ""  # 配置类路径
    _target_: str = ""  # 目标类路径

    def __post_init__(self):
        self._config_class_name_ = objtostr(self)

    def instantiate_from_config(self, *args, **kwargs) -> object:
        """Convert an object to a file path based on its module and class name."""
        modules_name, class_name = self._target_.split(":")
        modules = importlib.import_module(modules_name)
        assert hasattr(modules, class_name), f"Module {modules_name} has no attribute {class_name}"

        class_obj = getattr(modules, class_name)
        return class_obj(self, *args, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典

        Returns:
            Dict[str, Any]: 配置字典
        """
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """将配置序列化为JSON字符串

        Args:
            indent: JSON缩进

        Returns:
            str: JSON字符串
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_yaml(self) -> str:
        """将配置序列化为YAML字符串

        Returns:
            str: YAML字符串
        """
        return yaml.dump(self.to_dict(), allow_unicode=True, default_flow_style=False)

    def save(self, file_path: str, format: str = "json") -> None:
        """保存配置到文件

        Args:
            file_path: 文件路径
            format: 文件格式，支持 "json" 或 "yaml"

        Raises:
            ValueError: 不支持的格式
            ConfigValidationError: 配置验证失败
        """
        # 先验证配置
        # self.validate()

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "json":
            content = self.to_json()
        elif format.lower() == "yaml":
            content = self.to_yaml()
        else:
            raise ValueError(f"不支持的格式: {format}，支持 'json' 或 'yaml'")

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """从字典创建配置实例

        Args:
            data: 配置字典

        Returns:
            T: 配置实例
        """
        # 过滤掉未知字段
        '''
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)
        '''
        target = cls()
        if data["_config_class_name_"] != objtostr(target):
            target = instantiate_config(data["_config_class_name_"])
        dict_to_config(data, target)
        return target

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """从JSON字符串创建配置实例

        Args:
            json_str: JSON字符串

        Returns:
            T: 配置实例
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls: Type[T], yaml_str: str) -> T:
        """从YAML字符串创建配置实例

        Args:
            yaml_str: YAML字符串

        Returns:
            T: 配置实例
        """
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    @classmethod
    def load(cls: Type[T], file_path: str) -> T:
        """从文件加载配置

        Args:
            file_path: 文件路径

        Returns:
            T: 配置实例

        Raises:
            ValueError: 不支持的文件格式
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() in ['.json']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return cls.from_json(f.read())
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return cls.from_yaml(f.read())
        else:
            raise ValueError(f"不支持的配置文件格式: {file_path.suffix}")

    def merge(self, other: 'BaseConfig') -> 'BaseConfig':
        """合并另一个配置

        Args:
            other: 要合并的配置

        Returns:
            BaseConfig: 合并后的新配置实例

        Raises:
            ConfigMergeError: 配置合并失败
        """
        if type(self) != type(other):
            raise Error(
                f"配置类型不匹配: {type(self).__name__} vs {type(other).__name__}",
                config_type=type(self).__name__
            )

        # 创建新实例，使用other的非None值覆盖self的值
        merged_data = self.to_dict()
        other_data = other.to_dict()

        for key, value in other_data.items():
            if value is not None:
                merged_data[key] = value

        return type(self).from_dict(merged_data)

    def copy(self: T) -> T:
        """创建配置的深拷贝

        Returns:
            T: 新的配置实例
        """
        return type(self).from_dict(self.to_dict())

    def diff(self, other: 'BaseConfig') -> Dict[str, Dict[str, Any]]:
        """比较两个配置的差异

        Args:
            other: 要比较的配置

        Returns:
            Dict: 差异信息，包含 'added', 'removed', 'modified' 三个键
        """
        self_dict = self.to_dict()
        other_dict = other.to_dict()

        diff_result = {
            'added': {},
            'removed': {},
            'modified': {}
        }

        # 检查新增的字段
        for key in other_dict.keys() - self_dict.keys():
            diff_result['added'][key] = other_dict[key]

        # 检查删除的字段
        for key in self_dict.keys() - other_dict.keys():
            diff_result['removed'][key] = self_dict[key]

        # 检查修改的字段
        for key in self_dict.keys() & other_dict.keys():
            if self_dict[key] != other_dict[key]:
                diff_result['modified'][key] = {
                    'from': self_dict[key],
                    'to': other_dict[key]
                }

        return diff_result

    def __str__(self) -> str:
        """字符串表示"""
        return f"{type(self).__name__}({self.to_dict()})"

    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"{type(self).__name__}(version={self.version}, {self.to_dict()})"
