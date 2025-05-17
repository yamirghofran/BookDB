"use client"

import { useEffect, useState } from "react"
import { ArrowLeft } from "lucide-react"
import { Link } from "@tanstack/react-router"
import { books } from "@/lib/sample-data"
import { authors } from "@/lib/authors-data"
import BookCarousel from "@/components/book-carousel"
import type { Author } from "@/lib/types"
import type { Book } from "@/lib/types"

import { createFileRoute } from "@tanstack/react-router"

// Define the Route using createFileRoute
export const Route = createFileRoute("/authors/$id")({
  component: AuthorPage, // Your existing component
  // You can add loaders, error components, etc. here if needed
  // loader: async ({ params }) => { /* fetch data for the book id */ },
  // errorComponent: ({ error }) => <div>Error loading book: {error.message}</div>,
})

export default function AuthorPage({ params }: { params: { id: string } }) {
  const [author, setAuthor] = useState<Author | null>(null)
  const [authorBooks, setAuthorBooks] = useState<Book[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Find the author with the matching ID
    const foundAuthor = authors.find((a) => a.id === params.id)

    if (foundAuthor) {
      setAuthor(foundAuthor)

      // Find all books by this author
      const authorBooks = books.filter((book) => book.authors.some((name) => name === foundAuthor.name))

      setAuthorBooks(authorBooks)
    }

    setLoading(false)
  }, [params.id])

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-12">
        <div className="animate-pulse">
          <div className="h-8 w-40 bg-muted rounded mb-8"></div>
          <div className="h-12 w-1/2 bg-muted rounded mb-4"></div>
          <div className="h-6 w-1/3 bg-muted rounded mb-8"></div>
          <div className="h-4 w-full bg-muted rounded mb-2"></div>
          <div className="h-4 w-full bg-muted rounded mb-2"></div>
          <div className="h-4 w-3/4 bg-muted rounded mb-8"></div>
        </div>
      </div>
    )
  }

  if (!author) {
    return (
      <div className="container mx-auto px-4 py-12">
        <Link to="/" className="flex items-center text-primary mb-8 hover:underline">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to search
        </Link>
        <div className="text-center py-12">
          <h1 className="text-2xl font-bold mb-4">Author not found</h1>
          <p className="text-muted-foreground">The author you're looking for doesn't exist or has been removed.</p>
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
        {/* Author image */}
        <div className="w-full md:w-1/4 lg:w-1/5">
          <div className="aspect-square relative rounded-lg overflow-hidden shadow-md">
            <img
              src={author.imageUrl || "/placeholder.svg?height=300&width=300"}
              alt={`Photo of ${author.name}`}
              className="object-cover w-full h-full"
            />
          </div>
        </div>

        {/* Author details */}
        <div className="w-full md:w-3/4 lg:w-4/5">
          <h1 className="text-3xl font-bold mb-4">{author.name}</h1>
          {author.birthYear && (
            <p className="text-muted-foreground mb-4">
              {author.birthYear} {author.deathYear ? `- ${author.deathYear}` : ""}
            </p>
          )}
          {author.bio && (
            <div className="mb-6">
              <p className="text-muted-foreground leading-relaxed">{author.bio}</p>
            </div>
          )}
        </div>
      </div>

      {/* Author's books section */}
      {authorBooks.length > 0 ? (
        <div className="mt-12">
          <h2 className="text-2xl font-bold mb-6">Books by {author.name}</h2>
          <BookCarousel books={authorBooks} />
        </div>
      ) : (
        <div className="mt-12 text-center py-8 bg-muted/20 rounded-lg">
          <p className="text-muted-foreground">No books found for this author.</p>
        </div>
      )}
    </div>
  )
}
