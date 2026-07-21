
from sqlalchemy.orm import DeclarativeBase

from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import Float
from sqlalchemy import Boolean
from sqlalchemy import Text
from sqlalchemy import String
from sqlalchemy import ARRAY


class Base(DeclarativeBase):
    pass


### JIKAN MAL MANGA TABLES ###


class Manga(Base):

    __tablename__ = "manga"

    mal_id = Column(Integer, primary_key=True)

    title = Column(String)
    title_english = Column(String)

    status = Column(String)
    chapters = Column(Integer)
    volumes = Column(Integer)

    synopsis = Column(Text)
    background = Column(Text)

    genre_ids = Column(ARRAY(Integer))
    theme_ids = Column(ARRAY(Integer))

    author_ids = Column(ARRAY(Integer))
    serialization_ids = Column(ARRAY(Integer))

    # score = Column(Float)
    # scored_by = Column(Integer)
    # rank = Column(Integer)
    # popularity = Column(Integer)

    embedding = Column(ARRAY(Float), nullable=True)

    details_fetched = Column(Boolean, default=False)

class Themes(Base):
    
    __tablename__ = "themes"

    mal_id = Column(Integer, primary_key=True)
    name = Column(String)
    type = Column(String)

class Genres(Base):
    
    __tablename__ = "genres"

    mal_id = Column(Integer, primary_key=True)
    name = Column(String)
    type = Column(String)

class Authors(Base):
    __tablename__ = "authors"

    mal_id = Column(Integer, primary_key=True)
    name = Column(String)
    type = Column(String)

class Serializations(Base):
    __tablename__ = "serializations"

    mal_id = Column(Integer, primary_key=True)
    name = Column(String)
    type = Column(String)


### ANILIST USER TABLES ###


class Anilist_Users(Base):
    __tablename__ = "anilist_users"

    user_id = Column(Integer, primary_key=True)
    name = Column(String)
    score_type = Column(String)


class Anilist_User_Favorites(Base):
    __tablename__ = "anilist_user_favorites"

    user_id = Column(Integer, primary_key=True)
    mal_id  = Column(Integer, primary_key=True)

class Anilist_User_Manga(Base):
    __tablename__ = "anilist_user_manga"

    user_id = Column(Integer, primary_key=True)
    mal_id  = Column(Integer, primary_key=True)

    status = Column(String)
    score = Column(Float)
    score_normalized = Column(Float)
    progress = Column(Integer)
