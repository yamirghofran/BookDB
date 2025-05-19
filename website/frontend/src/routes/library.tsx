import { createFileRoute, Link } from "@tanstack/react-router";
import { useQueries } from "@tanstack/react-query";
import BookCard from "@/components/book-card";
import { useUserLibrary } from "@/contexts/UserLibraryContext";
import { fetchBookById } from "@/lib/api";
import type { Book } from "@/lib/types";

export const Route = createFileRoute("/library")({
  component: LibraryPage,
});

function LibraryPage() {
  const { libraryBookIds, anonymousUserId } = useUserLibrary();

  const bookQueries = useQueries({
    queries: libraryBookIds.map((bookId) => ({
      queryKey: ["book", bookId, anonymousUserId], // Add anonymousUserId to key
      queryFn: () => fetchBookById(bookId),
      enabled: !!anonymousUserId && !!bookId, // Ensure userId and bookId are present
    })),
  });

  const isLoading = bookQueries.some((query) => query.isLoading);
  const booksWithError = bookQueries.filter((query) => query.isError);
  const successfullyFetchedBooks: Book[] = bookQueries
    .filter((query) => query.isSuccess && query.data)
    .map((query) => query.data as Book)
    .filter((book): book is Book => book !== undefined); // Ensure book is not undefined

  if (!anonymousUserId) {
    return (
      <main className="container mx-auto px-4 py-8 text-center">
        <p>Loading user data...</p>
      </main>
    );
  }

  if (libraryBookIds.length === 0) {
    return (
      <main className="container mx-auto px-4 py-8 text-center">
        <h1 className="text-3xl font-bold mb-8">My Library</h1>
        <p className="text-xl text-muted-foreground">Your library is empty.</p>
        <p className="mt-4">
          <Link to="/" className="text-primary hover:underline">
            Browse books to add some!
          </Link>
        </p>
      </main>
    );
  }

  if (isLoading && successfullyFetchedBooks.length === 0 && libraryBookIds.length > 0) {
    return (
      <main className="container mx-auto px-4 py-8 text-center">
        <h1 className="text-3xl font-bold mb-8">My Library</h1>
        <p>Loading your liked books...</p>
      </main>
    );
  }

  if (booksWithError.length > 0 && successfullyFetchedBooks.length === 0) {
    // Only show full error state if no books could be loaded at all
    return (
      <main className="container mx-auto px-4 py-8 text-center">
        <h1 className="text-3xl font-bold mb-8">My Library</h1>
        <p className="text-red-500">Error fetching your liked books:</p>
        <ul>
          {booksWithError.map((query, index) => {
            const erroringBookId = libraryBookIds[bookQueries.findIndex(q => q === query)];
            return (
              <li key={erroringBookId || index}>
                Failed to load book (ID: {erroringBookId || 'unknown'}): {(query.error as Error)?.message || "Unknown error"}
              </li>
            );
          })}
        </ul>
      </main>
    );
  }

  return (
    <main className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-center mb-12">My Library</h1>
      {successfullyFetchedBooks.length > 0 ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {successfullyFetchedBooks.map((book: Book) => (
            <Link to={`/books/${book.id}` as any} key={book.id} className="h-full">
              <BookCard book={book} />
            </Link>
          ))}
        </div>
      ) : (
        !isLoading && (
          <div className="text-center py-12">
            <p className="text-xl text-muted-foreground">
              No liked books to display. It's possible there was an issue fetching them.
            </p>
             <p className="mt-4">
              <Link to="/" className="text-primary hover:underline">
                Browse books to add some!
              </Link>
            </p>
          </div>
        )
      )}
      {/* Optionally display partial errors if some books loaded but others failed */}
      {booksWithError.length > 0 && successfullyFetchedBooks.length > 0 && (
        <div className="mt-8 text-center text-sm text-orange-500">
          <p>Note: Some books in your library could not be loaded.</p>
        </div>
      )}
    </main>
  );
}
