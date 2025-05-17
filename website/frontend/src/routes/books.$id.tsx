"use client"

import { useEffect, useState } from "react"
import { ArrowLeft, MessageSquare } from "lucide-react"
import { Link } from "@tanstack/react-router"
import { books } from "@/lib/sample-data"
import { authors } from "@/lib/authors-data"
import BookCarousel from "@/components/book-carousel"
import ReviewCard from "@/components/review-card"
import AddReviewForm from "@/components/add-review-form"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import type { Book } from "@/lib/types"
import type { Author } from "@/lib/types"
import type { Review } from "@/lib/types"
import { createFileRoute } from "@tanstack/react-router"

// Define the Route using createFileRoute
export const Route = createFileRoute("/books/$id")({
  component: BookDetailsPage, // Your existing component
  // You can add loaders, error components, etc. here if needed
  // loader: async ({ params }) => { /* fetch data for the book id */ },
  // errorComponent: ({ error }) => <div>Error loading book: {error.message}</div>,
})

function BookDetailsPage() {
  const { id: bookId } = Route.useParams()
  const [book, setBook] = useState<Book | null>(null)
  const [bookAuthors, setBookAuthors] = useState<Author[]>([])
  const [similarBooks, setSimilarBooks] = useState<Book[]>([])
  const [loading, setLoading] = useState(true)
  const [reviews, setReviews] = useState<Review[]>([])
  const [isReviewFormOpen, setIsReviewFormOpen] = useState(false)

  useEffect(() => {
    // Find the book with the matching ID
    const foundBook = books.find((b) => b.id === bookId)

    if (foundBook) {
      setBook(foundBook)

      // Set reviews
      setReviews(foundBook.reviews || [])

      // Find author information
      const bookAuthorInfo = authors.filter((author) => foundBook.authors.includes(author.name))
      setBookAuthors(bookAuthorInfo)

      // Find similar books based on genre
      if (foundBook.genre && foundBook.genre.length > 0) {
        const similar = books
          .filter((b) => b.id !== foundBook.id && b.genre && b.genre.some((g) => foundBook.genre?.includes(g)))
          .slice(0, 8)

        setSimilarBooks(similar)
      }
    }

    setLoading(false)
  }, [bookId])

  const handleAddReview = (newReview: Omit<Review, "id" | "userId" | "date" | "rating" | "userAvatar">) => {
    const review: Review = {
      ...newReview,
      id: `r${Date.now()}`,
      userId: `u${Date.now()}`,
      date: new Date().toISOString().split("T")[0],
      rating: 5, // Default rating since we're not collecting it
      userAvatar: "/placeholder.svg?height=50&width=50", // Default avatar
    }

    setReviews([...reviews, review])
  }

  if (loading) {
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
        <div className="w-full md:w-1/3 lg:w-1/4">
          <div className="aspect-[2/3] relative rounded-lg overflow-hidden shadow-md">
            <img
              src={book.coverUrl || "/placeholder.svg"}
              alt={`Cover of ${book.title}`}
              className="object-cover w-full h-full"
            />
          </div>
        </div>

        {/* Book details */}
        <div className="w-full md:w-2/3 lg:w-3/4">
          <h1 className="text-3xl font-bold mb-2">{book.title}</h1>

          {/* Authors with links */}
          <div className="mb-6">
            <p className="text-xl text-muted-foreground">
              by{" "}
              {book.authors.map((authorName, index) => {
                const author = bookAuthors.find((a) => a.name === authorName)
                return (
                  <span key={authorName}>
                    {index > 0 && ", "}
                    {author ? (
                      <Link to={`/authors/${author.id}` as any} className="text-primary hover:underline">
                        {authorName}
                      </Link>
                    ) : (
                      authorName
                    )}
                  </span>
                )
              })}
            </p>
          </div>

          {book.description && (
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-3">Description</h2>
              <p className="text-muted-foreground leading-relaxed">{book.description}</p>
            </div>
          )}

          {book.genre && book.genre.length > 0 && (
            <div>
              <h2 className="text-xl font-semibold mb-3">Genres</h2>
              <div className="flex flex-wrap gap-2">
                {book.genre.map((genre) => (
                  <Badge key={genre} variant="secondary">{genre}</Badge>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Similar books section */}
      {similarBooks.length > 0 && (
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
