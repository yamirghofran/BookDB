-- +goose Up
-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; -- Needed for uuid_generate_v4()

-- Enum for user activity types
CREATE TYPE user_activity_type AS ENUM (
    'search',
    'view_book',
    'view_author',
    'add_to_library',
    'remove_from_library',
    'post_review',
    'update_review',
    'delete_review',
    'login',
    'register',
    'update_profile'
);

-- Users Table  
CREATE TABLE Users (  
   id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
   ncf_id INTEGER,
   name VARCHAR(255) NOT NULL,  
   email VARCHAR(255) UNIQUE NOT NULL,  
   created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,  
   updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP  
);  
  
-- Authors Table  
CREATE TABLE Authors (  
	 id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),  
	 name VARCHAR(255) NOT NULL,
	 average_rating NUMERIC,
	 ratings_count INTEGER,
	 created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,  
	 updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP  
);  
  
-- Genres Table  
CREATE TABLE Genres (  
	id SERIAL PRIMARY KEY,  
	name VARCHAR(100) UNIQUE NOT NULL  
);  
  
-- Books Table  
CREATE TABLE Books (  
   id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
   ncf_id INTEGER,
   goodreads_id BIGINT NOT NULL,
   goodreads_url TEXT,
   title VARCHAR(255) NOT NULL,
   description TEXT,
   publication_year BIGINT,
   cover_image_url TEXT,
   average_rating NUMERIC,
   ratings_count BIGINT,
   search_vector TSVECTOR,  
   created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,  
   updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP  
);  
  
-- Junction table for Books and Authors  
CREATE TABLE BookAuthors (  
	 book_id UUID NOT NULL REFERENCES Books(id) ON DELETE CASCADE,  
	 author_id UUID NOT NULL REFERENCES Authors(id) ON DELETE CASCADE,  
	 PRIMARY KEY (book_id, author_id)  
);  
  
-- Junction table for Books and Genres  
CREATE TABLE BookGenres (  
	book_id UUID NOT NULL REFERENCES Books(id) ON DELETE CASCADE,  
	genre_id INTEGER NOT NULL REFERENCES Genres(id) ON DELETE CASCADE,  
	PRIMARY KEY (book_id, genre_id)  
);  
  
-- Junction table for Similar Books  
CREATE TABLE SimilarBooks (  
	  book_id_1 UUID NOT NULL REFERENCES Books(id) ON DELETE CASCADE,  
	  book_id_2 UUID NOT NULL REFERENCES Books(id) ON DELETE CASCADE,  
	  PRIMARY KEY (book_id_1, book_id_2),  
	  CONSTRAINT check_different_books CHECK (book_id_1 <> book_id_2),  
	  CONSTRAINT check_ordered_pair CHECK (book_id_1 < book_id_2)  
);  
  
-- Reviews/Posts Table  
CREATE TABLE Reviews (  
	 id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),  
	 text TEXT NOT NULL,  
	 rating SMALLINT,  
	 book_id UUID NOT NULL REFERENCES Books(id) ON DELETE CASCADE,  
	 user_id UUID NOT NULL REFERENCES Users(id) ON DELETE CASCADE,  
	 created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,  
	 updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,  
	 CONSTRAINT rating_range CHECK (rating IS NULL OR (rating >= 1 AND rating <= 5))  
);  
  
-- User Library Table  
CREATE TABLE UserLibrary (  
	 user_id UUID NOT NULL REFERENCES Users(id) ON DELETE CASCADE,  
	 book_id UUID NOT NULL REFERENCES Books(id) ON DELETE CASCADE,  
	 added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,  
	 PRIMARY KEY (user_id, book_id)  
);  
  
-- User Activity Log Table  
CREATE TABLE ActivityLogs (  
	  id BIGSERIAL PRIMARY KEY,  
	  user_id UUID REFERENCES Users(id) ON DELETE SET NULL,  
	  activity_type user_activity_type NOT NULL,  
	  target_id UUID,  
	  details JSONB,  
	  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP  
);

-- Indexes (Remain the same)
CREATE INDEX idx_users_email ON Users(email);
CREATE INDEX idx_authors_name ON Authors(name);
CREATE INDEX idx_genres_name ON Genres(name);
CREATE INDEX idx_books_title ON Books(title);
CREATE INDEX idx_books_publication_year ON Books(publication_year);
CREATE INDEX idx_books_search_vector ON Books USING GIN(search_vector);
CREATE INDEX idx_bookauthors_author_id ON BookAuthors(author_id);
CREATE INDEX idx_bookgenres_genre_id ON BookGenres(genre_id);
CREATE INDEX idx_similarbooks_book_id_2 ON SimilarBooks(book_id_2);
CREATE INDEX idx_reviews_book_id ON Reviews(book_id);
CREATE INDEX idx_reviews_user_id ON Reviews(user_id);
CREATE INDEX idx_reviews_rating ON Reviews(rating);
CREATE INDEX idx_userlibrary_book_id ON UserLibrary(book_id);
CREATE INDEX idx_activitylogs_user_id ON ActivityLogs(user_id);
CREATE INDEX idx_activitylogs_activity_type ON ActivityLogs(activity_type);
CREATE INDEX idx_activitylogs_created_at ON ActivityLogs(created_at);

-- +goose StatementBegin
-- Trigger function to update the search_vector column in Books
CREATE OR REPLACE FUNCTION update_book_search_vector()
RETURNS TRIGGER AS $$
DECLARE
author_names TEXT;
BEGIN
SELECT string_agg(a.name, ' ')
INTO author_names
FROM Authors a
         JOIN BookAuthors ba ON a.id = ba.author_id
WHERE ba.book_id = NEW.id;

NEW.search_vector :=
        setweight(to_tsvector('english', coalesce(NEW.title,'')), 'A') ||
        setweight(to_tsvector('english', coalesce(NEW.description,'')), 'B') ||
        setweight(to_tsvector('english', coalesce(author_names,'')), 'C');
RETURN NEW;
END;
$$ LANGUAGE plpgsql;
-- +goose StatementEnd

-- Trigger to update search_vector when book is inserted or updated
CREATE TRIGGER tsvectorupdate_book
    BEFORE INSERT OR UPDATE ON Books
                         FOR EACH ROW EXECUTE FUNCTION update_book_search_vector();

-- +goose StatementBegin
-- Trigger function to update search_vector when BookAuthors link changes
CREATE OR REPLACE FUNCTION update_book_search_on_author_change()
RETURNS TRIGGER AS $$
DECLARE
v_book_id UUID;
BEGIN
    IF (TG_OP = 'DELETE') THEN
        v_book_id := OLD.book_id;
ELSE -- INSERT or UPDATE
        v_book_id := NEW.book_id;
END IF;

-- Trigger an update on the corresponding book row to recalculate the vector
UPDATE Books SET updated_at = CURRENT_TIMESTAMP WHERE id = v_book_id;

-- If the UPDATE changed book_id (unlikely but possible), update the old book too
IF (TG_OP = 'UPDATE' AND OLD.book_id <> NEW.book_id) THEN
UPDATE Books SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.book_id;
END IF;

RETURN NULL; -- Result is ignored since it's an AFTER trigger
END;
$$ LANGUAGE plpgsql;
-- +goose StatementEnd

-- Trigger to update search_vector when authors are added/removed/changed
CREATE TRIGGER tsvectorupdate_book_authors
    AFTER INSERT OR DELETE OR UPDATE ON BookAuthors
FOR EACH ROW EXECUTE FUNCTION update_book_search_on_author_change();


-- +goose Down
-- Drop triggers first
DROP TRIGGER IF EXISTS tsvectorupdate_book_authors ON BookAuthors;
DROP TRIGGER IF EXISTS tsvectorupdate_book ON Books;

-- Drop trigger functions
DROP FUNCTION IF EXISTS update_book_search_on_author_change();
DROP FUNCTION IF EXISTS update_book_search_vector();

-- Drop tables in reverse order of creation (considering dependencies)
DROP TABLE IF EXISTS ActivityLogs;
DROP TABLE IF EXISTS UserLibrary;
DROP TABLE IF EXISTS Reviews;
DROP TABLE IF EXISTS SimilarBooks;
DROP TABLE IF EXISTS BookGenres;
DROP TABLE IF EXISTS BookAuthors;
DROP TABLE IF EXISTS Books;
DROP TABLE IF EXISTS Genres;
DROP TABLE IF EXISTS Authors;
DROP TABLE IF EXISTS Users;

-- Drop custom types
DROP TYPE IF EXISTS user_activity_type;

-- Drop extensions created in this migration
DROP EXTENSION IF EXISTS "uuid-ossp";