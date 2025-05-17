"use client"

import { useState, useEffect } from "react"
import { ArrowLeft, MessageSquare, Star, Heart } from "lucide-react" // Added Star and Heart
import { Link } from "@tanstack/react-router"
import { useQuery, keepPreviousData } from "@tanstack/react-query" // Imported keepPreviousData
import { fetchBookById, fetchSimilarBooks, fetchUsersWithBookInLibrary } from "@/lib/api"
import BookCarousel from "@/components/book-carousel"
import ReviewCard from "@/components/review-card"
import AddReviewForm from "@/components/add-review-form"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import type { Book, Review, UserInLibrary, PaginatedUsersResponse } from "@/lib/types"
import { useUserLibrary } from "@/contexts/UserLibraryContext" // Import the hook
import { createFileRoute } from "@tanstack/react-router"

// Define the Route using createFileRoute
export const Route = createFileRoute("/books/$id")({
	component: BookDetailsPage,
})

function BookDetailsPage() {
	const { id: bookId } = Route.useParams()
	const { isBookLiked, addBookToLibrary, removeBookFromLibrary } = useUserLibrary()
	const [reviews, setReviews] = useState<Review[]>([])
	const [isReviewFormOpen, setIsReviewFormOpen] = useState(false)
	const [currentReviewPage, setCurrentReviewPage] = useState(1)
	const [reviewsTotalCount, setReviewsTotalCount] = useState(0)
	const reviewsPerPage = 6 // Display 6 reviews per page (fits 2 rows in a 3-col grid)

	const [currentUsersInLibraryPage, setCurrentUsersInLibraryPage] = useState(1)
	const usersInLibraryPerPage = 9 // 3 columns * 3 rows for example

	const {
	   data: book,
	   isLoading: isLoadingBook,
	   error: bookError,
	   refetch: refetchBook, // Added to refetch book data
	 } = useQuery<
	   Book, // TQueryFnData: Data type returned by queryFn
	   Error, // TError
	   Book, // TData: Type of 'data' in the result
	   readonly ["book", string, number] // TQueryKey: Type of the queryKey
	 >({
		queryKey: ["book", bookId, currentReviewPage] as const, // Use 'as const' for precise tuple type
		queryFn: () =>
			fetchBookById(bookId, {
				reviewsLimit: reviewsPerPage,
				reviewsOffset: (currentReviewPage - 1) * reviewsPerPage,
			}),
		enabled: !!bookId, // Only run query if bookId is available
		placeholderData: keepPreviousData, // Updated to keepPreviousData for TanStack Query v5+
	})

	const {
    data: similarBooks,
    // isLoading: isLoadingSimilar, // Can be used if needed
    // error: similarError, // Can be used if needed
  } = useQuery<Book[], Error>({
		queryKey: ["similarBooks", bookId],
		queryFn: () => fetchSimilarBooks(bookId),
		enabled: !!bookId, // Only run query if bookId is available
	})

	const {
		data: usersInLibraryData,
		isLoading: isLoadingUsersInLibrary,
		// error: usersInLibraryError, // Can be used if needed
	} = useQuery<PaginatedUsersResponse, Error>({
		queryKey: ["usersInLibrary", bookId, currentUsersInLibraryPage] as const,
		queryFn: () =>
			fetchUsersWithBookInLibrary(bookId, {
				limit: usersInLibraryPerPage,
				offset: (currentUsersInLibraryPage - 1) * usersInLibraryPerPage,
			}),
		enabled: !!bookId,
		placeholderData: keepPreviousData,
	})
		
		// Effect to set local reviews if book data includes them (currently it doesn't from API)
		// This part might need adjustment if/when reviews are fetched from backend
  useEffect(() => {
    if (book) {
      setReviews(book.reviews || []);
      setReviewsTotalCount(book.reviewsTotalCount || 0);
    } else {
      setReviews([]);
      setReviewsTotalCount(0);
    }
  }, [book]);

  const totalReviewPages = Math.ceil(reviewsTotalCount / reviewsPerPage)

  const handleAddReview = async (newReviewData: Omit<Review, "id" | "userId" | "date" | "rating" | "userAvatar" | "userName">) => {
    // For simplicity, this example adds locally and then refetches the current page of reviews.
    // A more robust solution would involve POSTing the review to the backend
    // and then refetching or optimistically updating.
    const review: Review = {
      ...newReviewData,
      id: `r${Date.now()}`,
      userId: `u${Date.now()}`,
      userName: "Anonymous User",
      date: new Date().toISOString().split("T")[0],
      rating: 5,
      userAvatar: "/placeholder.svg?height=50&width=50",
    };
    // Optimistically add to local state for immediate feedback (optional)
    // setReviews((prevReviews) => [...prevReviews, review]);

    // TODO: Implement actual API call to add review
    console.log("Adding review (locally for now):", review);

    // Refetch book data to get the latest reviews for the current page
    // This will also update total counts if the backend supports it after adding a review.
    // If the new review causes a new page to exist, you might want to navigate to it.
    // For now, just refetch current page.
    await refetchBook();
    setIsReviewFormOpen(false); // Close form after submission
  };


 if (isLoadingBook && !book) { // Show loading only if no previous data
		return (
			<div className="container mx-auto px-4 py-12">
				<div className="animate-pulse">
					<div className="h-8 w-40 bg-muted rounded mb-8"></div>
					<div className="flex flex-col md:flex-row gap-8">
						<div className="w-full md:w-1/3 h-96 bg-muted rounded"></div>
						<div className="w-full md:w-2/3">
							<div className="h-10 w-3/4 bg-muted rounded mb-4"></div>
							<div className="h-6 w-1/2 bg-muted rounded mb-8"></div>
							<div className="h-4 w-full bg-muted rounded mb-2"></div>
							<div className="h-4 w-full bg-muted rounded mb-2"></div>
							<div className="h-4 w-3/4 bg-muted rounded mb-8"></div>
							<div className="flex gap-2 mb-8">
								<div className="h-8 w-20 bg-muted rounded"></div>
								<div className="h-8 w-20 bg-muted rounded"></div>
							</div>
						</div>
					</div>
				</div>
			</div>
		)
	}

	if (bookError) {
		return (
			<div className="container mx-auto px-4 py-12 text-center">
				<p className="text-red-500">Error loading book: {bookError.message}</p>
				<Link to="/" className="text-primary hover:underline mt-4 inline-block">
					<ArrowLeft className="mr-2 h-4 w-4 inline" />
					Back to search
				</Link>
			</div>
		)
	}

	if (!book) {
		return (
			<div className="container mx-auto px-4 py-12">
				<Link to="/" className="flex items-center text-primary mb-8 hover:underline">
					<ArrowLeft className="mr-2 h-4 w-4" />
					Back to search
				</Link>
				<div className="text-center py-12">
					<h1 className="text-2xl font-bold mb-4">Book not found (ID: {bookId})</h1>
					<p className="text-muted-foreground">The book you're looking for doesn't exist or has been removed.</p>
				</div>
			</div>
		)
	}

	return (
		<div className="container mx-auto px-4 py-12">
			<Link to="/" className="flex items-center text-primary mb-8 hover:underline">
				<ArrowLeft className="mr-2 h-4 w-4" />
				Back to search
			</Link>

			<div className="flex flex-col md:flex-row gap-8 mb-12">
				{/* Book cover */}
				<div className="w-full md:w-1/3 lg:w-1/4 flex flex-col"> {/* Added flex flex-col */}
					<div className="aspect-[2/3] relative rounded-lg overflow-hidden shadow-md mb-4"> {/* Added margin-bottom */}
						<img
							src={book.coverUrl || "/placeholder.svg"}
							alt={`Cover of ${book.title}`}
							className="object-cover w-full h-full"
						/>
					</div>
					<Button
						className="w-full flex items-center justify-center gap-2"
						variant="outline"
						onClick={(e) => {
							e.preventDefault()
							e.stopPropagation()
							if (isBookLiked(book.id)) {
								removeBookFromLibrary(book.id)
							} else {
								addBookToLibrary(book.id)
							}
						}}
					>
						<Heart
							className={`h-5 w-5 transition-colors ${
								isBookLiked(book.id) ? "fill-red-500 text-red-500" : "text-muted-foreground"
							}`}
						/>
						<span>{isBookLiked(book.id) ? "Remove from Library" : "Add to Library"}</span>
					</Button>
				</div>

				{/* Book details */}
				<div className="w-full md:w-2/3 lg:w-3/4">
					<h1 className="text-3xl font-bold mb-2">{book.title}</h1>

					{/* Authors */}
					{book.authors && book.authors.length > 0 && (
						<div className="mb-4"> {/* Reduced mb slightly */}
							<p className="text-xl text-muted-foreground">
								by {book.authors.join(", ")}
							</p>
						</div>
					)}

					{/* Average Rating and Ratings Count */}
					<div className="flex items-center mb-6">
						{book.averageRating !== undefined && (
							<div className="flex items-center mr-4">
								<Star className="h-5 w-5 text-yellow-400 mr-1" />
								<span className="text-lg font-semibold">{book.averageRating.toFixed(1)}</span>
								{book.ratingsCount !== undefined && (
									<span className="text-sm text-muted-foreground ml-1">({book.ratingsCount} ratings)</span>
								)}
							</div>
						)}
						{/* Placeholder if only ratingsCount is available without averageRating */}
						{book.averageRating === undefined && book.ratingsCount !== undefined && (
							<p className="text-sm text-muted-foreground">{book.ratingsCount} ratings</p>
						)}
					</div>

					{/* Genres - Simplified: API currently returns genre: [] for individual book */}
					{book.genre && book.genre.length > 0 && (
						<div className="mb-6">
							<div className="flex flex-wrap gap-2">
								{book.genre.map((g: string) => ( // Explicitly typed 'g'
									<Badge key={g} variant="secondary">{g}</Badge>
								))}
							</div>
						</div>
					)}

					{book.description && (
						<div className="mb-6">
							<h2 className="text-xl font-semibold mb-3">Description</h2>
							<p className="text-muted-foreground leading-relaxed">{book.description}</p>
						</div>
					)}
				</div>
			</div>

			{/* Similar books section */}
			{similarBooks && similarBooks.length > 0 && (
				<div className="mb-16">
					<h2 className="text-2xl font-bold mb-6">Similar Books You Might Enjoy</h2>
					<BookCarousel books={similarBooks} />
				</div>
			)}

			{/* Reviews section - full width */}
			<div className="w-full">
				<div className="bg-muted/20 rounded-lg p-6">
					<div className="flex items-center justify-between mb-6">
						<h2 className="text-xl font-semibold flex items-center">
							<MessageSquare className="mr-2 h-5 w-5" />
							Reviews
						</h2>
						<Button onClick={() => setIsReviewFormOpen(true)}>Add Review</Button>
					</div>

					{reviews.length > 0 ? (
						<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
							{reviews.map((review) => (
								<ReviewCard key={review.id} review={review} />
							))}
						</div>
					) : (
						<div className="text-center py-8">
							<p className="text-muted-foreground mb-2">No reviews yet</p>
							<p className="text-sm text-muted-foreground">Be the first to share your thoughts!</p>
						</div>
					)}

					{/* Pagination for reviews */}
					{totalReviewPages > 1 && (
						<div className="mt-8 flex justify-center items-center space-x-4">
							<Button
								onClick={() => setCurrentReviewPage((prev) => Math.max(prev - 1, 1))}
								disabled={currentReviewPage === 1 || isLoadingBook}
							>
								Previous
							</Button>
							<span className="text-sm text-muted-foreground">
								Page {currentReviewPage} of {totalReviewPages}
							</span>
							<Button
								onClick={() => setCurrentReviewPage((prev) => Math.min(prev + 1, totalReviewPages))}
								disabled={currentReviewPage === totalReviewPages || isLoadingBook}
							>
								Next
							</Button>
						</div>
					)}
				</div>
			</div>

			{/* Users in Library Section */}
			<div className="w-full mt-12">
				<div className="bg-card border rounded-lg p-6">
					<h2 className="text-xl font-semibold mb-6">In {usersInLibraryData?.totalUsers || 0} Libraries</h2>
					{isLoadingUsersInLibrary && !usersInLibraryData && <p>Loading users...</p>}
					{/* {usersInLibraryError && <p className="text-red-500">Error loading users.</p>} */}
					{usersInLibraryData && usersInLibraryData.users.length > 0 ? (
						<>
							<div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
								{usersInLibraryData.users.map((user: UserInLibrary) => (
									<div key={user.id} className="border p-3 rounded-md bg-background shadow-sm">
										{/* Placeholder for avatar - replace with actual img if available */}
										<div className="w-10 h-10 bg-muted rounded-full mb-2 mx-auto"></div>
										<p className="text-sm font-medium text-center truncate">{user.name}</p>
									</div>
								))}
							</div>
							{/* Pagination for users in library */}
							{usersInLibraryData.totalUsers > usersInLibraryPerPage && (
								<div className="mt-8 flex justify-center items-center space-x-4">
									<Button
										onClick={() => setCurrentUsersInLibraryPage((prev) => Math.max(prev - 1, 1))}
										disabled={currentUsersInLibraryPage === 1 || isLoadingUsersInLibrary}
									>
										Previous
									</Button>
									<span className="text-sm text-muted-foreground">
										Page {currentUsersInLibraryPage} of {Math.ceil(usersInLibraryData.totalUsers / usersInLibraryPerPage)}
									</span>
									<Button
										onClick={() => setCurrentUsersInLibraryPage((prev) => Math.min(prev + 1, Math.ceil(usersInLibraryData.totalUsers / usersInLibraryPerPage)))}
										disabled={currentUsersInLibraryPage === Math.ceil(usersInLibraryData.totalUsers / usersInLibraryPerPage) || isLoadingUsersInLibrary}
									>
										Next
									</Button>
								</div>
							)}
						</>
					) : (
						!isLoadingUsersInLibrary && (
							<p className="text-muted-foreground text-center py-4">No users have this book in their library yet.</p>
						)
					)}
				</div>
			</div>


			{/* Review form modal */}
			<AddReviewForm
				bookId={book.id}
				isOpen={isReviewFormOpen}
				onClose={() => setIsReviewFormOpen(false)}
				onSubmit={handleAddReview}
			/>
		</div>
	)
}
