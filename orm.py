import json
from collections.abc import Iterable
import asyncio

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
                if isinstance(current_obj, ManyToManyManager):
                    current_obj = current_obj.query().all()
            except AttributeError:
                return False
            if current_obj is None:
                return False

        if isinstance(current_obj, (list, set, QuerySet)):  # Handle many-to-many relationships
            if isinstance(current_obj, QuerySet): current_obj = current_obj.all()
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

    def first(self):
        if self.count() > 0: return self.all()[0]
        return None

    def all(self):
        """Return all objects in the queryset."""
        return list(self.data)

    def values(self,*fields):
        if not fields: fields = list(self.model._get_fields().keys())
        result = []
        for obj in self.all():
            item = {}
            for field in fields:
                if "__" in field:
                    parts = field.split("__")
                    value = obj
                    for part in parts:
                        value = getattr(value, part, None)
                        if value == None: break
                    item[field] = value
                else:
                    item[field] = getattr(obj,field,None)
            result.append(item)
        return result

    def values_list(self, *fields, flat=False, named=False):
        if flat and len(fields) != 1:
            raise TypeError("flat=True requires exactly one field")
        values = self.values(*fields)
        if flat:
            return [item[fields[0]] for item in values]
        if named:
            from collections import namedtuple
            tuple_class = namedtuple('Row',fields)
            return [ tuple_class(**item) for item in values ]
        return [ tuple(item.values()) for item in values ]

    def delete(self):
        """Delete all objects in the queryset."""
        if self.model in BaseModel._registry:
            for instance in self.data:
                if instance in BaseModel._registry[self.model]:
                    BaseModel._registry[self.model].remove(instance)

    def count(self):
        return len(self.all())

    def __repr__(self):
        return f"QuerySet<{self.model.__name__}>({len(self.data)})"


class ManyToMany:

    _relations = {}

    """Descriptor for many-to-many relationships."""
    def __init__(self, related_class, related_name=None):
        self.related_class = related_class
        self.related_name = related_name  # Reverse lookup name for the related model
        self.name = None  # Field name in the defining model

    def contribute_to_class(self, cls, name):
        """Register the field with the model class and set the field name."""
        self.name = name  # Set the field name
        setattr(cls, name, self)  # Assign the descriptor to the class

        # Register the reverse relationship if a reverse lookup name is provided
        if self.related_name == None:
            self.related_name = cls.__name__.lower()
        if hasattr(self.related_class, self.related_name): return

        reverse_descriptor = ManyToMany(
            related_class=cls,
            related_name=self.name
        )
        reverse_descriptor._relations = self._relations
        setattr(self.related_class, self.related_name, reverse_descriptor)

    def __get__(self, instance, owner):
        if instance is None: return self
        return ManyToManyManager(instance, self.name, self.related_class, self._relations)

    def __set__(self, instance, value):
        raise AttributeError("Cannot directly assign to a ManyToMany field. Use the manager instead.")


class ManyToManyManager:
    """Manager for many-to-many relationships."""
    def __init__(self, instance, field_name, related_class, relations):
        self.instance = instance
        self.field_name = field_name
        self.related_class = related_class
        self.relations = relations

    def _get_reverse_manager(self, related_instance):
        for name, attr in type(related_instance).__dict__.items():
            if isinstance(attr, ManyToMany) and attr.related_class == type(self.instance):
                return getattr(related_instance, name)
        raise AttributeError(f"No reverse relationships found on {self.related_class.__name__}")

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
        return QuerySet(self.related_class, self.relations.get(self.instance, set()))

    def all(self):
        return self.query().all()

    def add(self, *values):
        """Add related objects."""
        for value in values:
            if not isinstance(value, self.related_class):
                raise TypeError(f"Expected instance of {self.related_class.__name__}")
            self.relations.setdefault(self.instance, set()).add(value)
            # Add the reverse relation if applicable
            reverse_manager = self._get_reverse_manager(value)
            reverse_manager.relations.setdefault(value, set()).add(self.instance)

    def remove(self, *values):
        """Remove related objects."""
        if self.instance in self.relations:
            for value in values:
                self.relations[self.instance].discard(value)
                # Remove the reverse relation if applicable
                if hasattr(value, self.field_name):
                    reverse_manager = self._get_reverse_manager(value)
                    if value in reverse_manager.relations:
                        reverse_manager.relations[value].discard(self.instance)

    def clear(self):
        """Clear all related objects."""
        if self.instance in self.relations:
            related_instances = list(self.relations[self.instance])
            del self.relations[self.instance]
            # Clear the reverse relations if applicable
            for related_instance in related_instances:
                reverse_manager = self._get_reverse_manager(related_instance)
                if related_instance in reverse_manager.relations:
                    reverse_manager.relations[related_instance].discard(self.instance)

    def __repr__(self):
        return f"ManyToManyManager<{self.related_class.__name__}>({len(self.query().all())})"

class RelatedManager:
    def __init__(self,source_class, related_class, related_objects):
        self.source_class = source_class
        self.related_class = related_class
        self.related_objects = related_objects

    def all(self):
        return self.query().all()

    def query(self):
        return QuerySet(self.related_class, self.related_objects)

    def first(self):
        return self.query().first()

    def __repr__(self):
        related_class = self.related_class.__name__
        return f"RelatedManger<{self.source_class.__name__}->{related_class}>({self.query().count()})"

class ForeignKey:
    """ForeignKey implementation with relationship-specific reverse lookups."""

    # We'll use a nested dict structure to track relationships by name
    _reverse_relations = {}  # Format: {related_class: {relationship_name: {instance: set(related_instances)}}}

    def __init__(self, related_class=None, related_name=None):
        self.related_class = related_class
        self.related_name = related_name
        self._data = {}  # Stores forward relations {instance: related_instance}
        self.relationship_name = None  # Will be set when contributing to class

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self._data.get(instance)

    def __set__(self, instance, value):
        if value is None:
            # Handle nulling the relationship
            if instance in self._data:
                old_value = self._data[instance]
                self._remove_reverse_relation(instance, old_value)
                del self._data[instance]
            return

        if not isinstance(value, self.related_class):
            raise TypeError(f"Expected instance of {self.related_class.__name__}, got {type(value).__name__}")

        # Remove any existing relationship
        if instance in self._data:
            old_value = self._data[instance]
            self._remove_reverse_relation(instance, old_value)

        # Set the new relationship
        self._data[instance] = value
        self._add_reverse_relation(instance, value)

    def _add_reverse_relation(self, instance, related_instance):
        """Add a reverse relationship tracking by relationship name."""
        if self.related_class not in self._reverse_relations:
            self._reverse_relations[self.related_class] = {}

        if self.relationship_name not in self._reverse_relations[self.related_class]:
            self._reverse_relations[self.related_class][self.relationship_name] = {}

        if related_instance not in self._reverse_relations[self.related_class][self.relationship_name]:
            self._reverse_relations[self.related_class][self.relationship_name][related_instance] = set()

        self._reverse_relations[self.related_class][self.relationship_name][related_instance].add(instance)

    def _remove_reverse_relation(self, instance, related_instance):
        """Remove a reverse relationship."""
        if (self.related_class in self._reverse_relations and
            self.relationship_name in self._reverse_relations[self.related_class] and
            related_instance in self._reverse_relations[self.related_class][self.relationship_name]):

            self._reverse_relations[self.related_class][self.relationship_name][related_instance].discard(instance)

            # Clean up empty sets
            if not self._reverse_relations[self.related_class][self.relationship_name][related_instance]:
                del self._reverse_relations[self.related_class][self.relationship_name][related_instance]

    def contribute_reverse_field(self, cls):
        """Set up the reverse relationship accessor."""
        if self.related_class == "self":
            self.related_class = cls

        # Use the field name as the relationship identifier
        self.relationship_name = self.related_name or f"{cls.__name__.lower()}_set"

        def getter(instance):
            related_objects = self.related_objects(instance)
            return RelatedManager(cls, self.related_class, related_objects)

        setattr(self.related_class, self.relationship_name, property(getter))

    def related_objects(self, instance):
        """Get objects related through this specific relationship."""
        return (self._reverse_relations
                .get(self.related_class, {})
                .get(self.relationship_name, {})
                .get(instance, set()))

class Field:
    """Simple field descriptor."""
    def __init__(self, default=None):
        self.default = default
        self._data = {}

    def __get__(self, instance, owner):
        return self._data.get(instance, self.default)

    def __set__(self, instance, value):
        self._data[instance] = value

class BaseModel:
    _registry = {}

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.__class__._registry.setdefault(self.__class__, []).append(self)
        self._dynamic_fields = {}
        self._loaded = False
        self._loading_lock = asyncio.Lock()

    async def load(self,db=None):
        if self._loaded == True: return self
        if not hasattr(self,'_load'):
            raise NotImplementedError(f"_load() not implemented for: {self.__class__.__name__}")
        if db == None: raise ValueError("db cannot be None for load() function")
        if hasattr(self,"unique_fields") and self.unique_fields != None:
            if any([getattr(self,k) == None for k in self.unique_fields]):
                msg = [
                    f"Unique Fields cannot be None when trying to load:",
                    f"{self.__class__.__name__}",
                    f"{self.to_dict(depth=1)}"
                ]
                raise ValueError("\n".join(msg))
        if db.exit == True: raise ValueError("Database connection closed")
        if db.exit == False and db.started != True: await db.start()
        return await self._load(db=db)

    @classmethod
    def get_or_create(cls,**kwargs):
        instance = cls.query().filter(**kwargs).first()
        if instance != None: return instance
        return cls(**kwargs)

        # if instance != None: return instance
        # return cls(**kwargs)

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
        for name, field in { k:v for k,v in cls.__dict__.items()}.items():
            if isinstance(field, ManyToMany):
                field.contribute_to_class(cls, name)
            elif isinstance(field, ForeignKey):
                field.contribute_reverse_field(cls)

    def add_field(self, field_name, field_type=Field, **kwargs):
        field = field_type(**kwargs)
        self._dynamic_fields[field_name] = field
        setattr(self, field_name, kwargs.get('default',None))

    def _get_fields(self):
        fields = {}
        for k,v in type(self).__dict__.items():
            if k.startswith("_"): continue
            if isinstance(v, (Field, ForeignKey, ManyToMany)):
                fields[k] = v
        try:
            for k, v in self._dynamic_fields.items():
                if k.startswith("_"): continue
                if isinstance(v, (Field, ForeignKey, ManyToMany)):
                    fields[k] = v
        except:
            pass
        # Other Properties
        for k in dir(self):
            if k.startswith("_"): continue
            v = getattr(self,k)
            if isinstance(v,RelatedManager):
                fields[k] = v
        return fields

    @classmethod
    def _get_class_fields(cls):
        obj = cls()
        fields = obj._get_fields()
        obj.delete()
        return fields

    def __setattr__(self, key, value):
        """Ensure attributes are stored in self.__dict__ properly."""
        super(BaseModel,self).__setattr__(key,value)
        self.__dict__[key] = value  # Store attributes explicitly in instance dictionary

    # Custom Getters and setters
    def __setitem__(self, key, value):
        if key not in self._get_fields().keys():
            raise AttributeError(f"Field {key} doesn't exist on Instance: {self}")
        self.__dict__[key] = value

    def __getitem__(self, key):
        if key not in self._get_fields().keys():
            raise AttributeError(f"Field {key} doesn't exist on Instance: {self}")
        return self.__dict__.get(key)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)
    # End custom getters and setters

    def __repr__(self):
        display = []
        fieldsDict = self._get_fields()
        numFields = len(fieldsDict)
        if numFields == 2:
            for k,v in fieldsDict.items():
                if isinstance(v,Field):
                    display.append(f"{k}:{getattr(self,k)}")
                elif isinstance(v,ForeignKey):
                    display.append(k)
                elif isinstance(v,ManyToMany):
                    items = getattr(self,k).query().all()
                    display.append(f"{k}({len(items)})")
                elif isinstance(v, QuerySet):
                    display.append(f"{k}({v.count()})")
                elif isinstance(v, RelatedManager):
                    display.append(f"{k}:{v}")
        else:
            display.append(f"..{numFields}")
        display = ";".join(display)
        return f"{self.__class__.__name__}({display})"

    def to_dict(self,depth=1):
        return Serializer(self).to_dict(depth=depth)

class Serializer:
    """Handles serialization of BaseModel objects, including nested relationships."""

    def __init__(self, obj):
        """
        Initialize the serializer with an object or iterable (QuerySet, list, set).
        :param obj: The object or iterable to serialize.
        :param depth: The depth limit for nested serialization (None = unlimited).
        """
        self.obj = obj
        self.depth = 10

    def serialize(self, obj, _current_depth=0):
        """
        Recursively serialize an object, respecting the depth limit.
        :param obj: The object to serialize.
        :param _current_depth: Tracks current depth level (internal use).
        :param fk: Should serialize the foreign keys or reverse_related properties.
        :return: Serialized representation of the object.
        """
        if self.depth is not None and _current_depth >= self.depth:
            return repr(obj)  # Stop recursion and return string representation

        if isinstance(obj, (list, set, QuerySet)):
            if not(hasattr(obj,"__iter__")): obj = obj.all()
            return [self.serialize(item, _current_depth) for item in obj]

        if isinstance(obj, BaseModel):  # Handle model objects
            data = {}
            fields = obj._get_fields()
            for field in fields:
                value = getattr(obj, field)
                if isinstance(value, QuerySet):
                    if _current_depth + 1 < self.depth:
                        data[field] = self.serialize(value, _current_depth + 1)
                elif isinstance(value, BaseModel):
                    if _current_depth + 1 < self.depth:
                        data[field] = self.serialize(value, _current_depth + 1)  # Recursive serialization
                elif isinstance(value,ManyToManyManager):
                    if _current_depth + 1 < self.depth:
                        data[field] = self.serialize(value.query().all(), _current_depth + 1)  # Recursive serialization
                elif isinstance(value, RelatedManager):
                    if _current_depth + 1 < self.depth:
                        data[field] = self.serialize(value.query().all(), _current_depth + 1)  # Recursive serialization
                else:
                    data[field] = value  # Primitive value (string, int, etc.)
            data["__type__"] = obj.__class__.__name__
            return data

        return obj  # Return primitive types as-is

    def to_dict(self,depth=10):
        """Convert serialized data to a dictionary."""
        self.depth = depth
        return self.serialize(self.obj)