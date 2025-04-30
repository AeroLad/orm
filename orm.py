import json
from collections.abc import Iterable

class Q:
    def __init__(self, **conditions):
        self.conditions = conditions

    def evaluate(self, obj, queryset):
        """Evaluate if an object matches the Q condition."""
        for key, value in self.conditions.items():
            if not queryset._evaluate_condition(obj, key, value):
                return False
        return True

    def __or__(self, other):
        return Q_OR(self, other)

    def __and__(self, other):
        return Q_AND(self, other)


class Q_OR(Q):
    def __init__(self, q1, q2):
        self.q1 = q1
        self.q2 = q2

    def evaluate(self, obj, queryset):
        return self.q1.evaluate(obj, queryset) or self.q2.evaluate(obj, queryset)


class Q_AND(Q):
    def __init__(self, q1, q2):
        self.q1 = q1
        self.q2 = q2

    def evaluate(self, obj, queryset):
        return self.q1.evaluate(obj, queryset) and self.q2.evaluate(obj, queryset)


class QuerySet:
    def __init__(self, model, data=None):
        self.model = model
        self.data = set(data) if data is not None else set()

    def _apply_lookup(self, value, lookup_type, target_value):
        """Apply a specific lookup type (e.g., exact, contains, in)."""
        if lookup_type == 'exact':
            return value == target_value
        elif lookup_type == 'in':
            return value in target_value
        elif lookup_type == 'contains':
            return isinstance(value, str) and target_value in value
        elif lookup_type == 'icontains':
            return isinstance(value, str) and target_value.lower() in value.lower()
        elif lookup_type == 'gt':
            return value > target_value
        elif lookup_type == 'gte':
            return value >= target_value
        elif lookup_type == 'lt':
            return value < target_value
        elif lookup_type == 'lte':
            return value <= target_value
        else:
            raise ValueError(f"Unsupported lookup type: {lookup_type}")

    def _evaluate_condition(self, obj, key, value):
        """Evaluate a single condition, handling '__' syntax."""
        parts = key.split('__')
        current_obj = obj
        lookup_type = 'exact'  # Default lookup type

        for i, part in enumerate(parts):
            if i == len(parts) - 1 and part in ['exact', 'in', 'contains', 'icontains', 'gt', 'gte', 'lt', 'lte']:
                lookup_type = part
                continue
            try:
                current_obj = getattr(current_obj, part, None)
                if callable(current_obj):  # Handle ManyToManyManager or other callable attributes
                    current_obj = current_obj.query().all()
            except AttributeError:
                return False
            if current_obj is None:
                return False

        if isinstance(current_obj, set):  # Handle many-to-many relationships
            if lookup_type == 'in':
                return any(val in current_obj for val in value)
            elif lookup_type == 'exact':
                return value in current_obj
            else:
                raise ValueError(f"Unsupported lookup type '{lookup_type}' for many-to-many relationships.")
        else:
            return self._apply_lookup(current_obj, lookup_type, value)

    def filter(self, *q_objects, **conditions):
        """Filter the queryset based on conditions."""
        results = self.data

        # Apply keyword conditions
        for key, value in conditions.items():
            results = {obj for obj in results if self._evaluate_condition(obj, key, value)}

        # Apply Q objects
        if q_objects:
            filtered_results = set()
            for obj in results:
                if any(q.evaluate(obj, self) for q in q_objects):
                    filtered_results.add(obj)
            results = filtered_results

        return QuerySet(self.model, results)

    def exclude(self, *q_objects, **conditions):
        """Exclude objects from the queryset based on conditions."""
        results = self.data

        # Apply keyword conditions
        for key, value in conditions.items():
            results = {obj for obj in results if not self._evaluate_condition(obj, key, value)}

        # Apply Q objects
        if q_objects:
            filtered_results = set()
            for obj in results:
                if all(not q.evaluate(obj, self) for q in q_objects):
                    filtered_results.add(obj)
            results = filtered_results

        return QuerySet(self.model, results)

    def order_by(self, field):
        """Sort the queryset by a field."""
        reverse = field.startswith('-')
        field = field.lstrip('-')
        return QuerySet(self.model, sorted(self.data, key=lambda obj: getattr(obj, field), reverse=reverse))

    def paginate(self, page=1, per_page=10):
        """Paginate the queryset."""
        start = (page - 1) * per_page
        end = start + per_page
        return list(self.all())[start:end]

    def all(self):
        """Return all objects in the queryset."""
        return list(self.data)

    def delete(self):
        """Delete all objects in the queryset."""
        if self.model in BaseModel._registry:
            for instance in self.data:
                if instance in BaseModel._registry[self.model]:
                    BaseModel._registry[self.model].remove(instance)

    def __repr__(self):
        return f"QuerySet<{self.model.__name__}>({len(self.data)})"


class BaseModel:
    _registry = {}

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.__class__._registry.setdefault(self.__class__, []).append(self)

    def delete(self):
        """Delete the instance from the registry."""
        if self.__class__ in self._registry and self in self._registry[self.__class__]:
            self._registry[self.__class__].remove(self)

    @classmethod
    def query(cls):
        """Return a QuerySet for the model."""
        return QuerySet(cls, cls._registry.get(cls, []))

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, field in cls.__dict__.items():
            if isinstance(field, ManyToMany):
                field.contribute_to_class(cls, name)

    def __setattr__(self, key, value):
        """Ensure attributes are stored in self.__dict__ properly."""
        super(BaseModel,self).__setattr__(key,value)
        self.__dict__[key] = value  # Store attributes explicitly in instance dictionary

    def __repr__(self):
        key = list(self.__dict__.keys())[0]
        return f"{self.__class__.__name__}({key}:{getattr(self,key)})"


class ManyToManyManager:
    """Manager for many-to-many relationships."""
    def __init__(self, instance, field_name, related_class, relations):
        self.instance = instance
        self.field_name = field_name
        self.related_class = related_class
        self.relations = relations

    def set(self, values):
        """
        Set multiple related objects at once.

        :param values: A QuerySet, list, or set of related instances.
        """
        if isinstance(values, QuerySet):
            values = values.all()  # Convert QuerySet to a set of instances
        elif not isinstance(values, (list, set)):
            raise TypeError("Values must be a QuerySet, list, or set of related instances.")

        # Clear existing relations and add the new ones
        self.clear()
        for value in values:
            if not isinstance(value, self.related_class):
                raise TypeError(f"Expected instance of {self.related_class.__name__}")
            self.add(value)

    def query(self):
        """Return a QuerySet for related objects."""
        related_instances = self.relations.get(self.instance, set())
        return QuerySet(self.related_class, related_instances)

    def add(self, *values):
        """Add related objects."""
        for value in values:
            if not isinstance(value, self.related_class):
                raise TypeError(f"Expected instance of {self.related_class.__name__}")
            self.relations.setdefault(self.instance, set()).add(value)
            # Add the reverse relation if applicable
            if hasattr(value, self.field_name):
                getattr(value, self.field_name).add(self.instance)

    def remove(self, *values):
        """Remove related objects."""
        if self.instance in self.relations:
            for value in values:
                self.relations[self.instance].discard(value)
                # Remove the reverse relation if applicable
                if hasattr(value, self.field_name):
                    getattr(value, self.field_name).remove(self.instance)

    def clear(self):
        """Clear all related objects."""
        if self.instance in self.relations:
            related_instances = list(self.relations[self.instance])
            del self.relations[self.instance]
            # Clear the reverse relations if applicable
            for related_instance in related_instances:
                if hasattr(related_instance, self.field_name):
                    getattr(related_instance, self.field_name).remove(self.instance)

    def __repr__(self):
        return f"ManyToManyManager<{self.related_class.__name__}>({len(self.query().all())})"


class ManyToMany:
    """Descriptor for many-to-many relationships."""
    def __init__(self, related_class, reverse_lookup=None):
        self.related_class = related_class
        self.reverse_lookup = reverse_lookup  # Reverse lookup name for the related model
        self._relations = {}  # {instance: set(related_instances)}
        self.name = None  # Field name in the defining model

    def contribute_to_class(self, cls, name):
        """Register the field with the model class and set the field name."""
        self.name = name  # Set the field name
        setattr(cls, name, self)  # Assign the descriptor to the class

        # Register the reverse relationship if a reverse lookup name is provided
        if self.reverse_lookup:
            reverse_field = ManyToMany(cls)  # Create a reverse ManyToMany field
            reverse_field.contribute_to_class(self.related_class, self.reverse_lookup)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return ManyToManyManager(instance, self.name, self.related_class, self._relations)

    def __set__(self, instance, value):
        raise AttributeError("Cannot directly assign to a ManyToMany field. Use the manager instead.")


class ForeignKey:
    """A simple ForeignKey implementation with reverse lookup support."""

    _reverse_relations = {}  # Tracks reverse relations {related_class: {related_instance: set(instances)}}

    def __init__(self, related_class):
        self.related_class = related_class
        self._data = {}

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self._data.get(instance)

    def __set__(self, instance, value):
        if not isinstance(value, self.related_class):
            raise TypeError(f"Expected instance of {self.related_class.__name__}, got {type(value).__name__}")

        # Store the relation
        self._data[instance] = value

        # Register reverse lookup
        if value not in self._reverse_relations.setdefault(self.related_class, {}):
            self._reverse_relations[self.related_class][value] = set()
        self._reverse_relations[self.related_class][value].add(instance)

    def related_objects(self, instance):
        """Retrieve all objects related to this instance via ForeignKey."""
        return self._reverse_relations.get(self.related_class, {}).get(instance, set())

    def all(self, instance):
        """Return QuerySet-like behavior for reverse lookup."""
        return QuerySet(type(instance), self.related_objects(instance))


class Field:
    """Simple field descriptor."""
    def __init__(self, default=None):
        self.default = default
        self._data = {}

    def __get__(self, instance, owner):
        return self._data.get(instance, self.default)

    def __set__(self, instance, value):
        self._data[instance] = value


class Serializer:
    """Handles serialization of BaseModel objects, including nested relationships."""

    def __init__(self, obj, depth=None):
        """
        Initialize the serializer with an object or iterable (QuerySet, list, set).
        :param obj: The object or iterable to serialize.
        :param depth: The depth limit for nested serialization (None = unlimited).
        """
        self.obj = obj
        self.depth = depth

    def get_fields(self, obj):
        """Get the fields of a model dynamically."""
        if hasattr(obj, "__dict__"):
            return ( key for key in obj.__class__.__dict__.keys() if not key.startswith('_') )
        return ()

    def serialize(self, obj, _current_depth=0):
        """
        Recursively serialize an object, respecting the depth limit.
        :param obj: The object to serialize.
        :param _current_depth: Tracks current depth level (internal use).
        :return: Serialized representation of the object.
        """
        if self.depth is not None and _current_depth >= self.depth:
            # if isinstance(obj,set): obj = list(obj)
            # if isinstance(obj,list) or isinstance(obj,tuple):
            #     if len(obj) == 0: return []
            return repr(obj)  # Stop recursion and return string representation

        if isinstance(obj, (list, set, QuerySet)):
            return [self.serialize(item, _current_depth) for item in obj]

        if isinstance(obj, BaseModel):  # Handle model objects
            data = {}
            for field in self.get_fields(obj):
                value = getattr(obj, field)
                if isinstance(value, BaseModel):
                    data[field] = self.serialize(value, _current_depth + 1)  # Recursive serialization
                elif isinstance(value,ManyToManyManager):
                    data[field] = self.serialize(value.query().all(), _current_depth + 1)  # Recursive serialization
                else:
                    data[field] = value  # Primitive value (string, int, etc.)

            return data

        return obj  # Return primitive types as-is

    def to_dict(self):
        """Convert serialized data to a dictionary."""
        return self.serialize(self.obj)

    def to_json(self):
        """Convert serialized data to a JSON string."""
        return json.dumps(self.to_dict(), indent=4)
