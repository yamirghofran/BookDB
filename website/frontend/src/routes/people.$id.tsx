import { useEffect } from "react"; // Added useEffect
import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, BookOpen, MessageSquare, Users, ThumbsUp } from "lucide-react";
import BookCarousel from "@/components/book-carousel";
import PersonReviewCard from "@/components/PersonReviewCard"; // Changed to PersonReviewCard
import { fetchPersonDetails, fetchSimilarUsers, fetchBookRecommendationsForUser } from "@/lib/api";
import type { PersonDetails, Person, Book as BookFromTypes, Review } from "@/lib/types";

export const Route = createFileRoute("/people/$id")({
  component: PersonDetailsPage,
});

function PersonDetailsPage() {
  const { id: userId } = Route.useParams();

  const {
    data: personDetails,
    isLoading,
    isError,
    error,
  } = useQuery<PersonDetails, Error>({
    queryKey: ["personDetails", userId],
    queryFn: () => fetchPersonDetails(userId),
    enabled: !!userId,
  });

  const { data: similarUsers } = useQuery<Person[], Error>({
    queryKey: ["similarUsers", userId],
    queryFn: () => fetchSimilarUsers(userId),
    enabled: !!userId,
  });

  const { data: recommendedBooksForUser } = useQuery<BookFromTypes[], Error>({
    queryKey: ["bookRecommendationsForUser", userId],
    queryFn: () => fetchBookRecommendationsForUser(userId),
    enabled: !!userId,
  });

  useEffect(() => {
    if (personDetails) {
      console.log("Person Details Loaded:", personDetails);
    }
  }, [personDetails]);

  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-12">
        <div className="animate-pulse">
          <div className="h-8 w-1/3 bg-muted rounded mb-4"></div>
          <div className="h-6 w-1/4 bg-muted rounded mb-10"></div>
          
          <div className="mb-12">
            <div className="h-7 w-1/5 bg-muted rounded mb-6"></div>
            <div className="flex space-x-4">
              <div className="min-w-[240px] h-80 bg-muted rounded"></div>
              <div className="min-w-[240px] h-80 bg-muted rounded hidden md:block"></div>
              <div className="min-w-[240px] h-80 bg-muted rounded hidden lg:block"></div>
            </div>
          </div>

          <div>
            <div className="h-7 w-1/5 bg-muted rounded mb-6"></div>
            <div className="space-y-4">
              <div className="h-24 bg-muted rounded"></div>
              <div className="h-24 bg-muted rounded"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="container mx-auto px-4 py-12 text-center">
        <p className="text-red-500">Error loading person details: {error?.message || "Unknown error"}</p>
        <Link to="/" className="text-primary hover:underline mt-4 inline-block">
          <ArrowLeft className="mr-2 h-4 w-4 inline" />
          Back to home
        </Link>
      </div>
    );
  }

  if (!personDetails) {
    return (
      <div className="container mx-auto px-4 py-12 text-center">
        <p>Person not found.</p>
        <Link to="/" className="text-primary hover:underline mt-4 inline-block">
          <ArrowLeft className="mr-2 h-4 w-4 inline" />
          Back to home
        </Link>
      </div>
    );
  }

  const { user, libraryBooks, userReviews } = personDetails;

  return (
    <div className="container mx-auto px-4 py-12">
      <Link to="/" className="flex items-center text-primary mb-8 hover:underline">
        <ArrowLeft className="mr-2 h-4 w-4" />
        Back to home
      </Link>

      <h1 className="text-4xl font-bold mb-2">{user.name}</h1>
      <p className="text-lg text-muted-foreground mb-10">Profile</p>

      {/* Library Books Section */}
      {libraryBooks && libraryBooks.length > 0 && (
        <div className="mb-16">
          <h2 className="text-2xl font-bold mb-6 flex items-center">
            <BookOpen className="mr-3 h-6 w-6" />
            {user.name}'s Library ({libraryBooks.length})
          </h2>
          <BookCarousel books={libraryBooks} />
        </div>
      )}
      {libraryBooks && libraryBooks.length === 0 && (
         <div className="mb-16">
          <h2 className="text-2xl font-bold mb-6 flex items-center">
            <BookOpen className="mr-3 h-6 w-6" />
            {user.name}'s Library
          </h2>
          <p className="text-muted-foreground">This user's library is currently empty.</p>
        </div>
      )}

      {/* Recommended Books for User Section */}
      {recommendedBooksForUser && recommendedBooksForUser.length > 0 && (
        <div className="my-16">
          <h2 className="text-2xl font-bold mb-6 flex items-center">
            <ThumbsUp className="mr-3 h-6 w-6" />
            Recommended Books for {user.name}
          </h2>
          <BookCarousel books={recommendedBooksForUser} />
        </div>
      )}

      {/* Similar Users Section */}
      {similarUsers && similarUsers.length > 0 && (
        <div className="my-16">
          <h2 className="text-2xl font-bold mb-6 flex items-center">
            <Users className="mr-3 h-6 w-6" />
            People with Similar Taste
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {similarUsers.map((simUser: Person) => (
              <Link key={simUser.id} to="/people/$id" params={{ id: simUser.id }} className="block p-3 border rounded-lg hover:bg-muted/50 transition-colors text-center bg-card shadow-sm">
                {/* Basic user display for now */}
                <div className="w-16 h-16 bg-muted rounded-full mb-2 mx-auto flex items-center justify-center text-xl font-semibold">
                  {simUser.name.substring(0,1).toUpperCase()}
                </div>
                <p className="text-sm font-medium truncate">{simUser.name}</p>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* User Reviews Section */}
      <div className="mt-16">
        <h2 className="text-2xl font-bold mb-6 flex items-center">
          <MessageSquare className="mr-3 h-6 w-6" />
          Reviews by {user.name} ({userReviews.length})
        </h2>
        {userReviews && userReviews.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6"> {/* Changed to grid layout */}
            {userReviews.map((review: Review) => (
              <PersonReviewCard key={review.id} review={review} /> // Used PersonReviewCard
            ))}
          </div>
        ) : (
          <p className="text-muted-foreground">{user.name} hasn't written any reviews yet.</p>
        )}
      </div>
    </div>
  );
}