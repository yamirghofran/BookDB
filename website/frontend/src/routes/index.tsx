import { createFileRoute } from "@tanstack/react-router";
import { useState, useMemo, useEffect } from "react";
import { Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import BookCard from "@/components/book-card";
import SearchBar from "@/components/search-bar";
import { fetchBooks, searchBooksAPI } from "@/lib/api"; // Import searchBooksAPI
import type { Book } from "@/lib/types";

export const Route = createFileRoute("/")({
	component: App,
});

const INITIAL_PAGE_LOAD_LIMIT = 100; // Fetch top 100 books
const SEARCH_RESULTS_LIMIT = 50;  // Limit for actual search results
const DEBOUNCE_DELAY = 500; // milliseconds

// Custom hook for debouncing
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}


function App() {
	const [searchQuery, setSearchQuery] = useState("");
  const debouncedSearchQuery = useDebounce(searchQuery, DEBOUNCE_DELAY);

  // Query for fetching the initial list of books to display on the page
  const {
    data: initialBooksData,
    isLoading: isLoadingInitialBooks,
    error: initialBooksError
  } = useQuery<Book[], Error>({
    queryKey: ["initial-books"],
    queryFn: () => {
      console.log(`Fetching initial books with limit: ${INITIAL_PAGE_LOAD_LIMIT}`);
      return fetchBooks({ limit: INITIAL_PAGE_LOAD_LIMIT });
    },
  });

  // Query for fetching search suggestions based on debounced search query
  const {
    data: suggestionData,
    // isLoading: isLoadingSuggestions, // Can be used if specific loading state for suggestions is needed
    // error: suggestionError, // Can be used if specific error state for suggestions is needed
  } = useQuery<Book[], Error>({
    queryKey: ["book-suggestions", debouncedSearchQuery],
    queryFn: async () => {
      const query = debouncedSearchQuery.trim();
      console.log(`Fetching suggestions for query: "${query}", limit: ${SEARCH_RESULTS_LIMIT}`);
      return searchBooksAPI({ query: query, limit: SEARCH_RESULTS_LIMIT });
    },
    enabled: debouncedSearchQuery.trim() !== "", // Only run when there's a debounced search query
    staleTime: 60 * 1000, // Consider suggestions fresh for 1 minute
    refetchOnWindowFocus: false, // Don't refetch suggestions on window focus
  });

  useEffect(() => {
    if (initialBooksData) {
      console.log("Initial books data loaded:", initialBooksData);
    }
    if (suggestionData) {
      console.log("Suggestion data loaded:", suggestionData);
    }
  }, [initialBooksData, suggestionData]);

  const displayedBooks = useMemo(() => {
    return initialBooksData || []; // Always display the initial set of books
  }, [initialBooksData]);

  const searchSuggestions = useMemo(() => {
    return suggestionData || []; // Use suggestions from the dedicated query
  }, [suggestionData]);

  if (isLoadingInitialBooks) {
    return (
      <main className="container mx-auto px-4 py-8 text-center">
        <p>Loading books...</p>
      </main>
    );
  }

  if (initialBooksError) {
    return (
      <main className="container mx-auto px-4 py-8 text-center">
        <p>Error fetching books: {initialBooksError.message}</p>
      </main>
    );
  }

  return (
    <main className="container mx-auto px-4 py-8">
      <div className="max-w-3xl mx-auto mb-12">
        <h1 className="text-3xl font-bold text-center mb-8">Find Your Next Book</h1>
        <SearchBar
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          suggestions={searchSuggestions.slice(0, 5)} // Use suggestions from the new query
        />
      </div>

      {displayedBooks.length > 0 ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {displayedBooks.map((book: Book) => (
            <Link to={`/books/${book.id}` as any} key={book.id}>
              <BookCard book={book} />
            </Link>
          ))}
        </div>
      ) : (
         !isLoadingInitialBooks && !initialBooksError && ( // Show if not loading and no error for initial load
          <div className="text-center py-12">
            <p className="text-xl text-muted-foreground">No books to display.</p>
          </div>
        )
      )}
    </main>
  )
}
