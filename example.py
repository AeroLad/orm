from orm import *
import json

class User(BaseModel):
    name = Field()

class Tag(BaseModel):
    name = Field()

class Post(BaseModel):
    title = Field()
    author = ForeignKey(User,related_name="posts")
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

post1.tags.set([tag_python,tag_ai])

tags = Tag.query().all()
print(tags)
posts = Post.query()
print(posts)
print(post1)