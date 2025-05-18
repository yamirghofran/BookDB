"use client"

import { useState, useRef, useEffect } from "react"
import { Link } from "@tanstack/react-router"; // Import Link
import { Search } from "lucide-react"
import { Input } from "@/components/ui/input"
import type { Book } from "@/lib/types"

interface SearchBarProps {
  searchQuery: string
  setSearchQuery: (query: string) => void
  suggestions: Book[]
}

export default function SearchBar({ searchQuery, setSearchQuery, suggestions }: SearchBarProps) {
  const [showSuggestions, setShowSuggestions] = useState(false)
  const suggestionsRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Close suggestions when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        suggestionsRef.current &&
        !suggestionsRef.current.contains(event.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(event.target as Node)
      ) {
        setShowSuggestions(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [])

  return (
    <div className="relative">
      <div className="relative">
        <Input
          ref={inputRef}
          type="text"
          placeholder="Search for books by title or author..."
          className="w-full pl-10 h-12 text-base"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onFocus={() => setShowSuggestions(true)}
        />
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-5 w-5" />
      </div>

      {showSuggestions && suggestions.length > 0 && searchQuery && (
        <div
          ref={suggestionsRef}
          className="absolute z-10 mt-1 w-full bg-background border rounded-md shadow-lg max-h-60 overflow-auto"
        >
          {suggestions.map((book) => (
            <Link
              to={`/books/${book.id}` as any}
              key={book.id}
              className="block hover:bg-muted" // Make Link block for full clickable area
              onClick={() => setShowSuggestions(false)} // Close suggestions on click
            >
              <div
                className="p-3 cursor-pointer flex items-center gap-3"
                // Removed onClick from div as Link handles navigation
              >
                <img
                  src={book.coverUrl || "/placeholder.svg"}
                  alt={book.title}
                  className="h-12 w-9 object-cover rounded"
                />
                <div>
                  <p className="font-medium">{book.title}</p>
                  <p className="text-sm text-muted-foreground">{book.authors.join(", ")}</p>
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
