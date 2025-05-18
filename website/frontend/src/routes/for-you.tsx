import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import BookCard from "@/components/book-card";
import { useUserLibrary } from "@/contexts/UserLibraryContext";
import { fetchAnonymousRecommendations } from "@/lib/api";
import type { Book } from "@/lib/types";

export const Route = createFileRoute("/for-you")({
  component: ForYouPage,
});

function ForYouPage() {
  const { libraryBookIds, anonymousUserId } = useUserLibrary();

  const {
    data: recommendedBooks,
    isLoading,
    isError,
    error,
  } = useQuery<Book[], Error>({
    queryKey: ["anonymousRecommendations", anonymousUserId, libraryBookIds],
    queryFn: () => {
      if (!anonymousUserId || libraryBookIds.length === 0) {
        // Don't fetch if no user or no liked books to base recommendations on
        return Promise.resolve([]);
      }
      return fetchAnonymousRecommendations(libraryBookIds);
    },
    enabled: !!anonymousUserId && libraryBookIds.length > 0, // Only run query if user and liked books exist
  });

  if (!anonymousUserId) {
    return (
      <main className="container mx-auto px-4 py-8 text-center">
        <p>Loading user data to get recommendations...</p>
      </main>
    );
  }

  if (libraryBookIds.length === 0 && !isLoading) {
    return (
      <main className="container mx-auto px-4 py-8 text-center">
        <h1 className="text-3xl font-bold mb-8">For You</h1>
        <p className="text-xl text-muted-foreground">
          Like some books in the <Link to="/library" className="text-primary hover:underline">library</Link> or on the <Link to="/" className="text-primary hover:underline">homepage</Link> to get personalized recommendations.
        </p>
      </main>
    );
  }

  if (isLoading) {
    return (
      <main className="container mx-auto px-4 py-8 text-center">
        <h1 className="text-3xl font-bold mb-8">For You</h1>
        <p>Generating your recommendations...</p>
      </main>
    );
  }

  if (isError) {
    return (
      <main className="container mx-auto px-4 py-8 text-center">
        <h1 className="text-3xl font-bold mb-8">For You</h1>
        <p className="text-red-500">Error fetching recommendations: {error?.message || "Unknown error"}</p>
      </main>
    );
  }

  return (
    <main className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-center mb-12">For You</h1>
      {recommendedBooks && recommendedBooks.length > 0 ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {recommendedBooks.map((book: Book) => (
            <Link to="/books/$id" params={{ id: book.id }} key={book.id} className="h-full"> {/* Modified Link */}
              <BookCard book={book} />
            </Link>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <p className="text-xl text-muted-foreground">
            No recommendations found for you at this time. Try liking more books!
          </p>
           <p className="mt-4">
            <Link to="/" className="text-primary hover:underline">
              Browse books
            </Link>
          </p>
        </div>
      )}
    </main>
  );
}