# Feature flags system for controlled rollout of capabilities
# Gates riskier capabilities behind explicit switches

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime


@dataclass
class FeatureFlag:
    """A feature flag with metadata for controlled rollout."""
    name: str
    enabled: bool = False
    description: str = ""
    rollout_percentage: float = 100.0  # 0-100, for gradual rollout
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    def is_enabled(self) -> bool:
        """Check if the feature is enabled."""
        if not self.enabled:
            return False
        # For gradual rollout, you could add percentage-based logic here
        return True
    
    def update(self, enabled: bool, **kwargs):
        """Update the feature flag."""
        self.enabled = enabled
        self.updated_at = datetime.now()
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class FeatureFlagManager:
    """Manages feature flags for controlled rollout."""
    
    def __init__(self):
        self._flags: Dict[str, FeatureFlag] = {}
        # Initialize default feature flags
        self._init_default_flags()
    
    def _init_default_flags(self):
        """Initialize default feature flags."""
        self.register_flag(
            "policy_aware_execution",
            description="Enable policy-aware tool execution",
            enabled=True
        )
        self.register_flag(
            "provenance_tracking",
            description="Enable execution provenance and audit trails",
            enabled=True
        )
        self.register_flag(
            "risky_tools",
            description="Allow risky tools that require explicit opt-in",
            enabled=False
        )
        self.register_flag(
            "consciousness_integration",
            description="Enable consciousness core integration",
            enabled=False
        )
        self.register_flag(
            "emotion_detection",
            description="Enable emotion detection in agent responses",
            enabled=False
        )
        self.register_flag(
            "memory_compaction",
            description="Enable automatic memory/session compaction",
            enabled=False
        )
    
    def register_flag(self, name: str, description: str = "", enabled: bool = False):
        """Register a new feature flag."""
        if name not in self._flags:
            self._flags[name] = FeatureFlag(
                name=name,
                description=description,
                enabled=enabled
            )
    
    def enable(self, name: str):
        """Enable a feature."""
        if name in self._flags:
            self._flags[name].update(enabled=True)
        else:
            raise ValueError(f"Feature flag '{name}' not found")
    
    def disable(self, name: str):
        """Disable a feature."""
        if name in self._flags:
            self._flags[name].update(enabled=False)
        else:
            raise ValueError(f"Feature flag '{name}' not found")
    
    def is_enabled(self, name: str) -> bool:
        """Check if a feature is enabled."""
        if name not in self._flags:
            return False
        return self._flags[name].is_enabled()
    
    def is_disabled(self, name: str) -> bool:
        """Check if a feature is disabled."""
        return not self.is_enabled(name)
    
    def get_flag(self, name: str) -> Optional[FeatureFlag]:
        """Get a feature flag."""
        return self._flags.get(name)
    
    def list_flags(self) -> Dict[str, FeatureFlag]:
        """List all feature flags."""
        return self._flags.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize feature flags."""
        return {
            name: {
                "enabled": flag.enabled,
                "description": flag.description,
                "metadata": flag.metadata
            }
            for name, flag in self._flags.items()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureFlagManager":
        """Deserialize feature flags."""
        manager = cls()
        for name, flag_data in data.items():
            if name in manager._flags:
                manager._flags[name].update(
                    enabled=flag_data.get("enabled", False),
                    metadata=flag_data.get("metadata", {})
                )
        return manager
