import { createFileRoute } from '@tanstack/react-router'
import { useState } from "react"
import { Link } from "@tanstack/react-router"
import { authors } from "@/lib/authors-data"
import { books } from "@/lib/sample-data"
import { Input } from "@/components/ui/input"
import { Search } from "lucide-react"

export const Route = createFileRoute('/authors')({
  component: AuthorsPage,
})

export default function AuthorsPage() {
  const [searchQuery, setSearchQuery] = useState("")

  // Filter authors based on search query
  const filteredAuthors =
    searchQuery.trim() === ""
      ? authors
      : authors.filter((author) => author.name.toLowerCase().includes(searchQuery.toLowerCase()))

  // Get book count for each author
  const getBookCount = (authorName: string) => {
    return books.filter((book) => book.authors.some((author) => author === authorName)).length
  }

  // Sort authors alphabetically by name
  const sortedAuthors = [...filteredAuthors].sort((a, b) => a.name.localeCompare(b.name))

  return (
    <div className="container mx-auto px-4 py-12">
      <h1 className="text-3xl font-bold text-center mb-12">Authors</h1>

      {/* Search bar */}
      <div className="max-w-md mx-auto mb-12 relative">
        <Input
          type="text"
          placeholder="Search authors..."
          className="pl-10"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-5 w-5" />
      </div>

      {filteredAuthors.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-12 gap-y-2">
          {sortedAuthors.map((author) => (
            <Link
              href={`/authors/${author.id}`}
              key={author.id}
              className="py-3 px-2 border-b hover:bg-muted/50 transition-colors flex items-center justify-between group"
            >
              <div className="flex items-center gap-3">
                <span className="font-medium group-hover:text-primary transition-colors">{author.name}</span>
              </div>
              <span className="text-sm text-muted-foreground">
                {getBookCount(author.name)} {getBookCount(author.name) === 1 ? "book" : "books"}
              </span>
            </Link>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <p className="text-xl text-muted-foreground">No authors found matching "{searchQuery}"</p>
        </div>
      )}
    </div>
  )
}
