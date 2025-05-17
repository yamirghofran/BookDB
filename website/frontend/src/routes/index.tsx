import { createFileRoute } from "@tanstack/react-router";
import { useState, useMemo } from "react";
import { Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import BookCard from "@/components/book-card";
import SearchBar from "@/components/search-bar";
import { fetchBooks } from "@/lib/api";
import type { Book } from "@/lib/types"; // Import the canonical Book type

export const Route = createFileRoute("/")({
	component: App,
});

function App() {
	const [searchQuery, setSearchQuery] = useState("");

  const { data: allBooks, isLoading, error } = useQuery<Book[], Error>({
    queryKey: ["books"],
    queryFn: fetchBooks,
  });

  // Log allBooks when it's available
  if (allBooks) {
    console.log("Books data in App component (allBooks):", allBooks);
  }

  const filteredBooks = useMemo(() => {
    if (!allBooks) return [];
    if (!searchQuery.trim()) return allBooks;

    const query = searchQuery.toLowerCase();
    return allBooks.filter(
      (book) =>
        book.title.toLowerCase().includes(query) ||
        (book.authors && book.authors.some((author) => author.toLowerCase().includes(query))),
    );
  }, [allBooks, searchQuery]);

  if (isLoading) {
    return (
      <main className="container mx-auto px-4 py-8 text-center">
        <p>Loading books...</p>
      </main>
    );
  }

  if (error) {
    return (
      <main className="container mx-auto px-4 py-8 text-center">
        <p>Error fetching books: {error.message}</p>
      </main>
    );
  }

  return (
    <main className="container mx-auto px-4 py-8">
      <div className="max-w-3xl mx-auto mb-12">
        <h1 className="text-3xl font-bold text-center mb-8">Find Your Next Book</h1>
        <SearchBar searchQuery={searchQuery} setSearchQuery={setSearchQuery} suggestions={filteredBooks.slice(0, 5)} />
      </div>

      {filteredBooks.length > 0 ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {filteredBooks.map((book) => (
            <Link to={`/books/${book.id}` as any} key={book.id}>
              <BookCard book={book} />
            </Link>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <p className="text-xl text-muted-foreground">No books found matching "{searchQuery}"</p>
        </div>
      )}
    </main>
  )
}
