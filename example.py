from orm import BaseModel, ManyToMany, ForeignKey, Field, Q

class User(BaseModel):
    name = Field()

class Tag(BaseModel):
    name = Field()

class Post(BaseModel):
    title = Field()
    author = ForeignKey(User)
    tags = ManyToMany(Tag)

# Create Users
alice = User(name="Alice")
bob = User(name="Bob")
steve = User(name="Steve")

# Create Tags
tag_python = Tag(name="Python")
tag_ai = Tag(name="AI")
tag_ml = Tag(name="Machine Learning")

# Create Posts
post1 = Post(title="Alice's AI Post", author=alice)
post2 = Post(title="Bob's Python Post", author=bob)
post3 = Post(title="ML Breakthrough", author=alice)
post4 = Post(title="Deep Learning", author=alice)
post5 = Post(title="Advanced Python", author=bob)

# Assign Tags
# post1.tags = {tag_ai, tag_ml}
# post2.tags = {tag_python}
# post3.tags = {tag_ml}
# post4.tags = {tag_ai}
# post5.tags = {tag_python}

# OR Query: Find posts tagged with AI **OR** Python
q = Post.query()
print(User.query().filter(Q(name="Alice") | Q(name="Bob") | Q(name="Steve")))
post1.tags.add(tag_ai)
print(post1.tags)
print(post2.tags.query())
# print(post1.tag)
# filtered_posts = Post.query().filter(Q(tags=tag_ai) | Q(tags=tag_python)).all()
# filtered_posts = Post.query().filter(tags=tag_ai).all()
# print([post.title for post in filtered_posts])
# Output: ["Alice's AI Post", "Bob's Python Post", "Deep Learning", "Advanced Python"]
# print(Post._registry)

# AND Query: Find posts tagged with AI **AND** ML
# filtered_posts = Post.query().filter(Q(tags=tag_ai) & Q(tags=tag_ml)).all()
# print([post.title for post in filtered_posts])
# Output: ["Alice's AI Post"]

# Paginate Results (page 1, 2 posts per page)
# paginated_posts = Post.query().paginate(page=1, per_page=2)
# print([post.title for post in paginated_posts])
# Output: ["Alice's AI Post", "Bob's Python Post"]
