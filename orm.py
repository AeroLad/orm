class Q:
    def __init__(self, **conditions):
        self.conditions = conditions

    def evaluate(self, obj):
        """Check if an object matches the Q condition."""
        for key, value in self.conditions.items():
            if isinstance(value, set):  # OR condition for many-to-many
                if not any(val in getattr(obj, key, []) for val in value):
                    return False
            elif isinstance(value, BaseModel):  # Foreign key exact match
                if getattr(obj, key, None) != value:
                    return False
            else:  # Regular field match
                if getattr(obj, key, None) != value:
                    return False
        return True

    def __or__(self, other):
        """OR operator (|) between Q objects."""
        return Q_OR(self, other)

    def __and__(self, other):
        """AND operator (&) between Q objects."""
        return Q_AND(self, other)


class Q_OR(Q):
    def __init__(self, q1, q2):
        self.q1 = q1
        self.q2 = q2

    def evaluate(self, obj):
        return self.q1.evaluate(obj) or self.q2.evaluate(obj)


class Q_AND(Q):
    def __init__(self, q1, q2):
        self.q1 = q1
        self.q2 = q2

    def evaluate(self, obj):
        return self.q1.evaluate(obj) and self.q2.evaluate(obj)


class QuerySet:
    def __init__(self, model, data):
        self.model = model
        self.data = set(data)  # Use set for faster lookups

    def filter(self, *q_objects, **conditions):
        """Supports Q objects (OR/AND) and traditional keyword filtering."""
        results = self.data

        # Process traditional keyword filters
        for key, value in conditions.items():
            if isinstance(value, set):  # OR condition (many-to-many)
                results = {obj for obj in results if any(val in getattr(obj, key, []) for val in value)}
            elif isinstance(value, BaseModel):  # Foreign key match
                results = {obj for obj in results if getattr(obj, key, None) == value}
            else:  # Regular field match
                results = {obj for obj in results if getattr(obj, key, None) == value}

        # Process Q objects
        if q_objects:
            filtered_results = set()
            for obj in results:
                if any(q.evaluate(obj) for q in q_objects):
                    filtered_results.add(obj)
            results = filtered_results

        return QuerySet(self.model, results)

    def delete(self):
        """Deletes all instances in the queryset."""
        if self.model in BaseModel._registry:
            for instance in self.data:
                if instance in BaseModel._registry[self.model]:
                    BaseModel._registry[self.model].remove(instance)

    def order_by(self, field):
        """Sort results."""
        reverse = field.startswith("-")
        field = field.lstrip("-")
        return QuerySet(self.model, sorted(self.data, key=lambda obj: getattr(obj, field), reverse=reverse))

    def paginate(self, page=1, per_page=10):
        """Paginate results."""
        start = (page - 1) * per_page
        end = start + per_page
        return self.all()[start:end]

    def all(self):
        return self.data  # Convert back to list for final results

    def __repr__(self):
        return f"QuerySet<{self.model.__name__}>({len(self.data)})"


class BaseModel:
    _registry = {}

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.__class__._registry.setdefault(self.__class__, []).append(self)

    def delete(self):
        """Delete a single instance from the registry."""
        if self.__class__ in self._registry and self in self._registry[self.__class__]:
            self._registry[self.__class__].remove(self)

    @classmethod
    def query(cls):
        """Returns a QuerySet for advanced queries."""
        return QuerySet(cls, cls._registry.get(cls, []))

    def __repr__(self):
        return f"{self.__class__.__name__}"


class ManyToManyManager:
    """A helper class to manage ManyToMany relationships per instance."""
    def __init__(self, instance, many_to_many_field):
        self.instance = instance
        self.many_to_many_field = many_to_many_field

    def query(self):
        """Fetch related objects as a QuerySet."""
        return QuerySet(self.many_to_many_field.related_class,
                        self.many_to_many_field._relations.get(self.instance, set()))

    def add(self, value):
        """Add a related object to the ManyToMany relationship."""
        if not isinstance(value, self.many_to_many_field.related_class):
            raise TypeError(f"Expected instance of {self.many_to_many_field.related_class.__name__}")
        self.many_to_many_field._relations.setdefault(self.instance, set()).add(value)

    def remove(self, value):
        """Remove a related object from the ManyToMany relationship."""
        if self.instance in self.many_to_many_field._relations:
            self.many_to_many_field._relations[self.instance].discard(value)

    def clear(self):
        """Remove all related objects from this instance."""
        if self.instance in self.many_to_many_field._relations:
            self.many_to_many_field._relations[self.instance].clear()

    def __repr__(self):
        return f"ManyToManyManager<{self.many_to_many_field.related_class.__name__}>({len(self.query().all())})"


class ManyToMany:
    """Optimized many-to-many relationship using dictionary indexing."""

    def __init__(self, related_class):
        self.related_class = related_class
        self._relations = {}  # {instance: set(related_instances)}

    def __get__(self, instance, owner):
        if instance is None: return self
        return ManyToManyManager(instance, self)

    def query(self):
        """Fetch related objects lazily."""
        return QuerySet(self.related_class, self._relations.get(self.related_class, set()))

    def __set__(self, values):
        """Set multiple related objects efficiently."""
        if not isinstance(values, (list, set)):
            raise TypeError(f"Expected a list or set of {self.related_class.__name__} instances")
        if not all(isinstance(v, self.related_class) for v in values):
            raise TypeError(f"Expected instances of {self.related_class.__name__}")
        self._relations[self.related_class] = set(values)

    def add(self, value):
        """Add a relationship efficiently."""
        instance = type(value)
        if not isinstance(value, self.related_class):
            raise TypeError(f"Expected {self.related_class.__name__}")
        self._relations.setdefault(self.related_class, set()).add(value)
        return True

    def remove(self, value):
        """Remove a relationship."""
        instance = type(value)
        if instance in self._relations and value in self._relations[self.related_class]:
            self._relations[self.related_class].remove(value)
            return True
        return False

    def clear(self):
        """Remove all related objects."""
        self.data[self.related_class].clear()


class ForeignKey:
    """ForeignKey field for one-to-many relationships."""

    def __init__(self, related_class):
        self.related_class = related_class
        self._data = {}

    def __get__(self, instance, owner):
        """Fetch the related object."""
        return self._data.get(instance, None)

    def __set__(self, instance, value):
        """Set a foreign key reference."""
        if not isinstance(value, self.related_class):
            raise TypeError(f"Expected instance of {self.related_class.__name__}")
        self._data[instance] = value


class Field:
    """Simple field descriptor."""
    def __init__(self, default=None):
        self.default = default
        self._data = {}

    def __get__(self, instance, owner):
        return self._data.get(instance, self.default)

    def __set__(self, instance, value):
        self._data[instance] = value
