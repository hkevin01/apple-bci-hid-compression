"""Input mapping system implementations."""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

# Import from our gesture recognition
try:
    from .gesture_recognition import GestureEvent, GestureType
except ImportError:
    # Fallback definitions if import fails
    class GestureType(Enum):
        CLICK = "click"
        DOUBLE_CLICK = "double_click"
        RIGHT_CLICK = "right_click"
        SCROLL_UP = "scroll_up"
        SCROLL_DOWN = "scroll_down"
        SWIPE_LEFT = "swipe_left"
        SWIPE_RIGHT = "swipe_right"
        HOLD = "hold"
        TAP = "tap"
        MOVE = "move"


class InputType(Enum):
    """Types of input actions."""
    MOUSE_CLICK = "mouse_click"
    MOUSE_DOUBLE_CLICK = "mouse_double_click"
    MOUSE_RIGHT_CLICK = "mouse_right_click"
    MOUSE_MOVE = "mouse_move"
    MOUSE_SCROLL = "mouse_scroll"
    MOUSE_DRAG = "mouse_drag"

    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"
    KEY_COMBINATION = "key_combination"

    TOUCH_TAP = "touch_tap"
    TOUCH_DOUBLE_TAP = "touch_double_tap"
    TOUCH_LONG_PRESS = "touch_long_press"
    TOUCH_SWIPE = "touch_swipe"
    TOUCH_PINCH = "touch_pinch"
    TOUCH_ROTATE = "touch_rotate"

    SYSTEM_COMMAND = "system_command"
    APPLICATION_SHORTCUT = "application_shortcut"
    ACCESSIBILITY_ACTION = "accessibility_action"


class MappingType(Enum):
    """Types of mapping configurations."""
    FIXED = "fixed"
    CONFIGURABLE = "configurable"
    CONTEXT_AWARE = "context_aware"
    ADAPTIVE = "adaptive"


@dataclass
class InputAction:
    """Represents an input action to be performed."""
    action_type: InputType
    parameters: Dict[str, Any] = field(default_factory=dict)
    modifiers: List[str] = field(default_factory=list)
    target: Optional[str] = None  # Target application/window
    delay: float = 0.0  # Delay before execution

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'action_type': self.action_type.value,
            'parameters': self.parameters,
            'modifiers': self.modifiers,
            'target': self.target,
            'delay': self.delay
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InputAction':
        """Create from dictionary."""
        return cls(
            action_type=InputType(data['action_type']),
            parameters=data.get('parameters', {}),
            modifiers=data.get('modifiers', []),
            target=data.get('target'),
            delay=data.get('delay', 0.0)
        )


@dataclass
class MappingRule:
    """Represents a mapping rule from gesture to input action."""
    gesture_type: GestureType
    input_action: InputAction
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'gesture_type': self.gesture_type.value,
            'input_action': self.input_action.to_dict(),
            'conditions': self.conditions,
            'priority': self.priority,
            'enabled': self.enabled
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MappingRule':
        """Create from dictionary."""
        return cls(
            gesture_type=GestureType(data['gesture_type']),
            input_action=InputAction.from_dict(data['input_action']),
            conditions=data.get('conditions', {}),
            priority=data.get('priority', 0),
            enabled=data.get('enabled', True)
        )


class InputMapper(ABC):
    """Abstract base class for input mapping."""

    @abstractmethod
    def map_gesture(self, gesture: GestureEvent) -> Optional[InputAction]:
        """Map a gesture to an input action."""
        pass

    @abstractmethod
    def add_mapping(self, rule: MappingRule) -> bool:
        """Add a new mapping rule."""
        pass

    @abstractmethod
    def remove_mapping(self, gesture_type: GestureType,
                      conditions: Optional[Dict[str, Any]] = None) -> bool:
        """Remove a mapping rule."""
        pass


class FixedInputMapper(InputMapper):
    """Fixed mapping implementation with predefined rules."""

    def __init__(self):
        self.mappings: Dict[GestureType, InputAction] = {}
        self._load_default_mappings()

    def _load_default_mappings(self):
        """Load default fixed mappings."""
        # Basic mouse controls
        self.mappings[GestureType.CLICK] = InputAction(
            action_type=InputType.MOUSE_CLICK,
            parameters={'button': 'left'}
        )

        self.mappings[GestureType.DOUBLE_CLICK] = InputAction(
            action_type=InputType.MOUSE_DOUBLE_CLICK,
            parameters={'button': 'left'}
        )

        self.mappings[GestureType.RIGHT_CLICK] = InputAction(
            action_type=InputType.MOUSE_RIGHT_CLICK,
            parameters={'button': 'right'}
        )

        # Scrolling
        self.mappings[GestureType.SCROLL_UP] = InputAction(
            action_type=InputType.MOUSE_SCROLL,
            parameters={'direction': 'up', 'amount': 3}
        )

        self.mappings[GestureType.SCROLL_DOWN] = InputAction(
            action_type=InputType.MOUSE_SCROLL,
            parameters={'direction': 'down', 'amount': 3}
        )

        # Swiping as navigation
        self.mappings[GestureType.SWIPE_LEFT] = InputAction(
            action_type=InputType.KEY_COMBINATION,
            parameters={'keys': ['cmd', 'left']},
            modifiers=['command']
        )

        self.mappings[GestureType.SWIPE_RIGHT] = InputAction(
            action_type=InputType.KEY_COMBINATION,
            parameters={'keys': ['cmd', 'right']},
            modifiers=['command']
        )

        # Hold as special action
        self.mappings[GestureType.HOLD] = InputAction(
            action_type=InputType.ACCESSIBILITY_ACTION,
            parameters={'action': 'voice_control_toggle'}
        )

        print(f"Loaded {len(self.mappings)} fixed mappings")

    def map_gesture(self, gesture: GestureEvent) -> Optional[InputAction]:
        """Map gesture to fixed input action."""
        if gesture.gesture_type in self.mappings:
            action = self.mappings[gesture.gesture_type]
            print(f"Fixed: Mapped {gesture.gesture_type.value} -> {action.action_type.value}")
            return action

        print(f"Fixed: No mapping found for {gesture.gesture_type.value}")
        return None

    def add_mapping(self, rule: MappingRule) -> bool:
        """Add fixed mapping (replaces existing)."""
        self.mappings[rule.gesture_type] = rule.input_action
        print(f"Added fixed mapping: {rule.gesture_type.value}")
        return True

    def remove_mapping(self, gesture_type: GestureType,
                      conditions: Optional[Dict[str, Any]] = None) -> bool:
        """Remove fixed mapping."""
        if gesture_type in self.mappings:
            del self.mappings[gesture_type]
            print(f"Removed fixed mapping: {gesture_type.value}")
            return True
        return False

    def get_all_mappings(self) -> Dict[GestureType, InputAction]:
        """Get all current mappings."""
        return self.mappings.copy()


class ConfigurableInputMapper(InputMapper):
    """Configurable mapping with user-defined rules and profiles."""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.mapping_rules: List[MappingRule] = []
        self.profiles: Dict[str, List[MappingRule]] = {}
        self.current_profile = "default"

        # Load configuration
        if config_file:
            self.load_configuration(config_file)
        else:
            self._create_default_profile()

    def _create_default_profile(self):
        """Create default configurable profile."""
        default_rules = [
            MappingRule(
                gesture_type=GestureType.CLICK,
                input_action=InputAction(
                    action_type=InputType.MOUSE_CLICK,
                    parameters={'button': 'left'}
                ),
                priority=1
            ),
            MappingRule(
                gesture_type=GestureType.DOUBLE_CLICK,
                input_action=InputAction(
                    action_type=InputType.MOUSE_DOUBLE_CLICK,
                    parameters={'button': 'left'}
                ),
                priority=1
            ),
            MappingRule(
                gesture_type=GestureType.SCROLL_UP,
                input_action=InputAction(
                    action_type=InputType.MOUSE_SCROLL,
                    parameters={'direction': 'up', 'amount': 5}
                ),
                priority=1
            ),
            MappingRule(
                gesture_type=GestureType.SCROLL_DOWN,
                input_action=InputAction(
                    action_type=InputType.MOUSE_SCROLL,
                    parameters={'direction': 'down', 'amount': 5}
                ),
                priority=1
            )
        ]

        self.profiles["default"] = default_rules
        self.mapping_rules = default_rules.copy()
        print("Created default configurable profile")

    def map_gesture(self, gesture: GestureEvent) -> Optional[InputAction]:
        """Map gesture using configurable rules."""
        # Find matching rules
        matching_rules = [
            rule for rule in self.mapping_rules
            if rule.gesture_type == gesture.gesture_type and rule.enabled
        ]

        # Apply conditions filtering
        valid_rules = []
        for rule in matching_rules:
            if self._check_conditions(gesture, rule.conditions):
                valid_rules.append(rule)

        if not valid_rules:
            return None

        # Sort by priority (higher priority first)
        valid_rules.sort(key=lambda r: r.priority, reverse=True)

        selected_rule = valid_rules[0]
        print(f"Configurable: Mapped {gesture.gesture_type.value} -> " +
              f"{selected_rule.input_action.action_type.value} (priority: {selected_rule.priority})")

        return selected_rule.input_action

    def _check_conditions(self, gesture: GestureEvent, conditions: Dict[str, Any]) -> bool:
        """Check if gesture meets rule conditions."""
        if not conditions:
            return True

        # Check confidence threshold
        if 'min_confidence' in conditions:
            if gesture.confidence < conditions['min_confidence']:
                return False

        # Check duration range
        if 'duration_range' in conditions:
            duration_range = conditions['duration_range']
            if not (duration_range[0] <= gesture.duration <= duration_range[1]):
                return False

        # Check time of day
        if 'time_range' in conditions:
            current_hour = time.localtime().tm_hour
            time_range = conditions['time_range']
            if not (time_range[0] <= current_hour <= time_range[1]):
                return False

        # Check gesture parameters
        if 'parameters' in conditions:
            for param_key, expected_value in conditions['parameters'].items():
                if gesture.parameters.get(param_key) != expected_value:
                    return False

        return True

    def add_mapping(self, rule: MappingRule) -> bool:
        """Add configurable mapping rule."""
        try:
            self.mapping_rules.append(rule)
            # Also add to current profile
            if self.current_profile in self.profiles:
                self.profiles[self.current_profile].append(rule)

            print(f"Added configurable mapping: {rule.gesture_type.value}")
            return True
        except Exception as e:
            print(f"Failed to add mapping: {e}")
            return False

    def remove_mapping(self, gesture_type: GestureType,
                      conditions: Optional[Dict[str, Any]] = None) -> bool:
        """Remove configurable mapping rule."""
        initial_count = len(self.mapping_rules)

        # Remove from main rules
        self.mapping_rules = [
            rule for rule in self.mapping_rules
            if not (rule.gesture_type == gesture_type and
                   (conditions is None or rule.conditions == conditions))
        ]

        # Remove from current profile
        if self.current_profile in self.profiles:
            self.profiles[self.current_profile] = [
                rule for rule in self.profiles[self.current_profile]
                if not (rule.gesture_type == gesture_type and
                       (conditions is None or rule.conditions == conditions))
            ]

        removed_count = initial_count - len(self.mapping_rules)
        if removed_count > 0:
            print(f"Removed {removed_count} configurable mappings for {gesture_type.value}")
            return True

        return False

    def create_profile(self, profile_name: str, rules: Optional[List[MappingRule]] = None):
        """Create a new mapping profile."""
        if rules is None:
            rules = []

        self.profiles[profile_name] = rules
        print(f"Created profile: {profile_name}")

    def switch_profile(self, profile_name: str) -> bool:
        """Switch to a different mapping profile."""
        if profile_name in self.profiles:
            self.current_profile = profile_name
            self.mapping_rules = self.profiles[profile_name].copy()
            print(f"Switched to profile: {profile_name}")
            return True

        print(f"Profile not found: {profile_name}")
        return False

    def save_configuration(self, file_path: str) -> bool:
        """Save configuration to file."""
        try:
            config_data = {
                'profiles': {
                    name: [rule.to_dict() for rule in rules]
                    for name, rules in self.profiles.items()
                },
                'current_profile': self.current_profile
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)

            print(f"Saved configuration to: {file_path}")
            return True
        except Exception as e:
            print(f"Failed to save configuration: {e}")
            return False

    def load_configuration(self, file_path: str) -> bool:
        """Load configuration from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # Load profiles
            self.profiles = {}
            for profile_name, rules_data in config_data.get('profiles', {}).items():
                rules = [MappingRule.from_dict(rule_data) for rule_data in rules_data]
                self.profiles[profile_name] = rules

            # Set current profile
            current_profile = config_data.get('current_profile', 'default')
            if current_profile in self.profiles:
                self.current_profile = current_profile
                self.mapping_rules = self.profiles[current_profile].copy()

            print(f"Loaded configuration from: {file_path}")
            return True
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            return False


class ContextAwareInputMapper(InputMapper):
    """Context-aware mapping that adapts based on current application and environment."""

    def __init__(self):
        self.mapping_rules: List[MappingRule] = []
        self.context_handlers: Dict[str, Callable] = {}
        self.current_context: Dict[str, Any] = {}

        # Register default context handlers
        self._register_context_handlers()
        self._create_context_aware_mappings()

    def _register_context_handlers(self):
        """Register context detection handlers."""
        self.context_handlers['application'] = self._detect_active_application
        self.context_handlers['window_title'] = self._detect_window_title
        self.context_handlers['input_focus'] = self._detect_input_focus
        self.context_handlers['screen_region'] = self._detect_screen_region
        self.context_handlers['time_of_day'] = self._detect_time_context

    def _create_context_aware_mappings(self):
        """Create context-aware mapping rules."""
        # Web browser context
        browser_click = MappingRule(
            gesture_type=GestureType.CLICK,
            input_action=InputAction(
                action_type=InputType.MOUSE_CLICK,
                parameters={'button': 'left'}
            ),
            conditions={
                'application': ['Safari', 'Chrome', 'Firefox', 'Edge'],
                'context_priority': 'high'
            },
            priority=5
        )

        # Text editor context
        editor_swipe_left = MappingRule(
            gesture_type=GestureType.SWIPE_LEFT,
            input_action=InputAction(
                action_type=InputType.KEY_COMBINATION,
                parameters={'keys': ['cmd', 'z']},  # Undo
                modifiers=['command']
            ),
            conditions={
                'application': ['TextEdit', 'VS Code', 'Xcode', 'Sublime'],
                'input_focus': 'text_field'
            },
            priority=6
        )

        # Video player context
        video_swipe_right = MappingRule(
            gesture_type=GestureType.SWIPE_RIGHT,
            input_action=InputAction(
                action_type=InputType.KEY_PRESS,
                parameters={'key': 'space'}  # Play/pause
            ),
            conditions={
                'application': ['QuickTime', 'VLC', 'YouTube'],
                'window_title_contains': ['video', 'player', 'movie']
            },
            priority=7
        )

        # Gaming context
        game_hold = MappingRule(
            gesture_type=GestureType.HOLD,
            input_action=InputAction(
                action_type=InputType.KEY_PRESS,
                parameters={'key': 'shift'}
            ),
            conditions={
                'application_category': 'games',
                'fullscreen': True
            },
            priority=8
        )

        # Accessibility context (evening hours)
        evening_click = MappingRule(
            gesture_type=GestureType.CLICK,
            input_action=InputAction(
                action_type=InputType.ACCESSIBILITY_ACTION,
                parameters={'action': 'speak_selection'},
                delay=0.1  # Small delay for voice feedback
            ),
            conditions={
                'time_range': [18, 23],  # 6 PM to 11 PM
                'accessibility_mode': True
            },
            priority=9
        )

        self.mapping_rules = [
            browser_click, editor_swipe_left, video_swipe_right,
            game_hold, evening_click
        ]

        print(f"Created {len(self.mapping_rules)} context-aware mappings")

    def map_gesture(self, gesture: GestureEvent) -> Optional[InputAction]:
        """Map gesture using context-aware rules."""
        # Update current context
        self._update_context()

        # Find matching rules
        matching_rules = [
            rule for rule in self.mapping_rules
            if rule.gesture_type == gesture.gesture_type and rule.enabled
        ]

        # Apply context filtering
        context_rules = []
        for rule in matching_rules:
            if self._check_context_conditions(rule.conditions):
                context_rules.append(rule)

        if not context_rules:
            # Fallback to basic mapping
            return self._get_fallback_mapping(gesture)

        # Sort by priority and context relevance
        context_rules.sort(key=lambda r: (r.priority, self._calculate_context_score(r)),
                          reverse=True)

        selected_rule = context_rules[0]
        print(f"Context-aware: Mapped {gesture.gesture_type.value} -> " +
              f"{selected_rule.input_action.action_type.value} " +
              f"(context: {self.current_context.get('application', 'unknown')})")

        return selected_rule.input_action

    def _update_context(self):
        """Update current context information."""
        new_context = {}

        for context_type, handler in self.context_handlers.items():
            try:
                new_context[context_type] = handler()
            except Exception as e:
                print(f"Context handler error ({context_type}): {e}")

        self.current_context = new_context

    def _detect_active_application(self) -> str:
        """Detect currently active application."""
        # Mock implementation - in real system would use system APIs
        mock_apps = ['Safari', 'VS Code', 'TextEdit', 'QuickTime', 'Terminal']
        import random
        return random.choice(mock_apps)

    def _detect_window_title(self) -> str:
        """Detect current window title."""
        # Mock implementation
        mock_titles = ['Untitled', 'Document', 'video.mp4', 'browser tab', 'terminal']
        import random
        return random.choice(mock_titles)

    def _detect_input_focus(self) -> str:
        """Detect current input focus type."""
        # Mock implementation
        focus_types = ['text_field', 'button', 'menu', 'canvas', 'web_page']
        import random
        return random.choice(focus_types)

    def _detect_screen_region(self) -> Dict[str, int]:
        """Detect current screen region of interest."""
        # Mock implementation
        return {'x': 100, 'y': 200, 'width': 800, 'height': 600}

    def _detect_time_context(self) -> Dict[str, Any]:
        """Detect time-based context."""
        current_time = time.localtime()
        return {
            'hour': current_time.tm_hour,
            'day_of_week': current_time.tm_wday,
            'is_weekend': current_time.tm_wday >= 5
        }

    def _check_context_conditions(self, conditions: Dict[str, Any]) -> bool:
        """Check if current context meets rule conditions."""
        if not conditions:
            return True

        # Check application condition
        if 'application' in conditions:
            required_apps = conditions['application']
            current_app = self.current_context.get('application', '')
            if current_app not in required_apps:
                return False

        # Check window title condition
        if 'window_title_contains' in conditions:
            required_words = conditions['window_title_contains']
            current_title = self.current_context.get('window_title', '').lower()
            if not any(word.lower() in current_title for word in required_words):
                return False

        # Check input focus condition
        if 'input_focus' in conditions:
            required_focus = conditions['input_focus']
            current_focus = self.current_context.get('input_focus', '')
            if current_focus != required_focus:
                return False

        # Check time range condition
        if 'time_range' in conditions:
            time_range = conditions['time_range']
            time_context = self.current_context.get('time_of_day', {})
            current_hour = time_context.get('hour', 12)
            if not (time_range[0] <= current_hour <= time_range[1]):
                return False

        return True

    def _calculate_context_score(self, rule: MappingRule) -> float:
        """Calculate relevance score for rule in current context."""
        score = 0.0
        conditions = rule.conditions

        # Application match score
        if 'application' in conditions:
            required_apps = conditions['application']
            current_app = self.current_context.get('application', '')
            if current_app in required_apps:
                score += 2.0

        # Context priority boost
        if conditions.get('context_priority') == 'high':
            score += 1.0

        # Time relevance
        if 'time_range' in conditions:
            score += 0.5

        return score

    def _get_fallback_mapping(self, gesture: GestureEvent) -> Optional[InputAction]:
        """Get fallback mapping when no context rules match."""
        # Simple fallback mappings
        fallback_map = {
            GestureType.CLICK: InputAction(InputType.MOUSE_CLICK, {'button': 'left'}),
            GestureType.DOUBLE_CLICK: InputAction(InputType.MOUSE_DOUBLE_CLICK, {'button': 'left'}),
            GestureType.SCROLL_UP: InputAction(InputType.MOUSE_SCROLL, {'direction': 'up', 'amount': 3}),
            GestureType.SCROLL_DOWN: InputAction(InputType.MOUSE_SCROLL, {'direction': 'down', 'amount': 3})
        }

        fallback_action = fallback_map.get(gesture.gesture_type)
        if fallback_action:
            print(f"Context-aware: Using fallback mapping for {gesture.gesture_type.value}")

        return fallback_action

    def add_mapping(self, rule: MappingRule) -> bool:
        """Add context-aware mapping rule."""
        self.mapping_rules.append(rule)
        print(f"Added context-aware mapping: {rule.gesture_type.value}")
        return True

    def remove_mapping(self, gesture_type: GestureType,
                      conditions: Optional[Dict[str, Any]] = None) -> bool:
        """Remove context-aware mapping rule."""
        initial_count = len(self.mapping_rules)

        self.mapping_rules = [
            rule for rule in self.mapping_rules
            if not (rule.gesture_type == gesture_type and
                   (conditions is None or rule.conditions == conditions))
        ]

        removed_count = initial_count - len(self.mapping_rules)
        if removed_count > 0:
            print(f"Removed {removed_count} context-aware mappings for {gesture.gesture_type.value}")
            return True

        return False

    def get_current_context(self) -> Dict[str, Any]:
        """Get current context information."""
        return self.current_context.copy()

    def add_context_handler(self, context_type: str, handler: Callable) -> bool:
        """Add custom context detection handler."""
        try:
            self.context_handlers[context_type] = handler
            print(f"Added context handler: {context_type}")
            return True
        except Exception as e:
            print(f"Failed to add context handler: {e}")
            return False


class MultiModalInputMapper:
    """Multi-modal mapper that combines different mapping approaches."""

    def __init__(self):
        self.mappers: Dict[MappingType, InputMapper] = {
            MappingType.FIXED: FixedInputMapper(),
            MappingType.CONFIGURABLE: ConfigurableInputMapper(),
            MappingType.CONTEXT_AWARE: ContextAwareInputMapper()
        }

        self.mapping_priority = [
            MappingType.CONTEXT_AWARE,
            MappingType.CONFIGURABLE,
            MappingType.FIXED
        ]

        self.fallback_enabled = True
        self.gesture_history: List[GestureEvent] = []
        self.mapping_statistics: Dict[str, int] = {}

    def map_gesture(self, gesture: GestureEvent) -> Optional[InputAction]:
        """Map gesture using multi-modal approach."""
        # Add to history
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > 100:
            self.gesture_history.pop(0)

        # Try mappers in priority order
        for mapping_type in self.mapping_priority:
            mapper = self.mappers[mapping_type]

            try:
                action = mapper.map_gesture(gesture)
                if action:
                    # Update statistics
                    mapper_name = mapping_type.value
                    self.mapping_statistics[mapper_name] = (
                        self.mapping_statistics.get(mapper_name, 0) + 1
                    )

                    print(f"Multi-modal: Mapped using {mapper_name} mapper")
                    return action
            except Exception as e:
                print(f"Mapper error ({mapping_type.value}): {e}")
                continue

        print(f"Multi-modal: No mapping found for {gesture.gesture_type.value}")
        return None

    def add_mapping(self, mapping_type: MappingType, rule: MappingRule) -> bool:
        """Add mapping to specific mapper."""
        if mapping_type in self.mappers:
            return self.mappers[mapping_type].add_mapping(rule)
        return False

    def remove_mapping(self, mapping_type: MappingType, gesture_type: GestureType,
                      conditions: Optional[Dict[str, Any]] = None) -> bool:
        """Remove mapping from specific mapper."""
        if mapping_type in self.mappers:
            return self.mappers[mapping_type].remove_mapping(gesture_type, conditions)
        return False

    def set_mapping_priority(self, priority_order: List[MappingType]):
        """Set priority order for mapping types."""
        # Validate all types are available
        if all(mt in self.mappers for mt in priority_order):
            self.mapping_priority = priority_order
            print(f"Updated mapping priority: {[mt.value for mt in priority_order]}")
            return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get mapping statistics."""
        total_mappings = sum(self.mapping_statistics.values())

        stats = {
            'total_gestures_mapped': total_mappings,
            'recent_gestures': len(self.gesture_history),
            'mapper_usage': self.mapping_statistics.copy(),
            'mapper_percentages': {}
        }

        # Calculate percentages
        if total_mappings > 0:
            for mapper, count in self.mapping_statistics.items():
                stats['mapper_percentages'][mapper] = (count / total_mappings) * 100

        return stats

    def get_mapper(self, mapping_type: MappingType) -> Optional[InputMapper]:
        """Get specific mapper instance."""
        return self.mappers.get(mapping_type)
