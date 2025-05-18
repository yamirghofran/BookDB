import { useState, useRef } from "react"
import { Link } from "@tanstack/react-router" // Added Link import
import { ChevronLeft, ChevronRight } from "lucide-react"
import BookCard from "./book-card"
import type { Book } from "@/lib/types"

interface BookCarouselProps {
  books: Book[]
}

export default function BookCarousel({ books }: BookCarouselProps) {
  const carouselRef = useRef<HTMLDivElement>(null)
  const [scrollPosition, setScrollPosition] = useState(0)

  const scroll = (direction: "left" | "right") => {
    if (!carouselRef.current) return

    const container = carouselRef.current
    const scrollAmount = container.clientWidth * 0.8

    if (direction === "left") {
      container.scrollBy({ left: -scrollAmount, behavior: "smooth" })
    } else {
      container.scrollBy({ left: scrollAmount, behavior: "smooth" })
    }

    // Update scroll position after scrolling
    setTimeout(() => {
      if (carouselRef.current) {
        setScrollPosition(carouselRef.current.scrollLeft)
      }
    }, 300)
  }

  // Handle scroll event to update button visibility
  const handleScroll = () => {
    if (carouselRef.current) {
      setScrollPosition(carouselRef.current.scrollLeft)
    }
  }

  const showLeftButton = scrollPosition > 0
  const showRightButton = carouselRef.current
    ? scrollPosition < carouselRef.current.scrollWidth - carouselRef.current.clientWidth - 10
    : true

  return (
    <div className="relative">
      {/* Left scroll button */}
      {showLeftButton && (
        <button
          onClick={() => scroll("left")}
          className="absolute left-0 top-1/2 -translate-y-1/2 z-10 bg-background/80 backdrop-blur-sm rounded-full p-2 shadow-md"
          aria-label="Scroll left"
        >
          <ChevronLeft className="h-6 w-6" />
        </button>
      )}

      {/* Carousel container */}
      <div ref={carouselRef} className="flex overflow-x-auto gap-4 pb-4 scrollbar-hide snap-x" onScroll={handleScroll}>
        {books.map((book) => (
          <div key={book.id} className="min-w-[240px] snap-start">
            <Link to="/books/$id" params={{ id: book.id }} className="block h-full"> {/* Modified Link */}
              <BookCard book={book} />
            </Link>
          </div>
        ))}
      </div>

      {/* Right scroll button */}
      {showRightButton && (
        <button
          onClick={() => scroll("right")}
          className="absolute right-0 top-1/2 -translate-y-1/2 z-10 bg-background/80 backdrop-blur-sm rounded-full p-2 shadow-md"
          aria-label="Scroll right"
        >
          <ChevronRight className="h-6 w-6" />
        </button>
      )}
    </div>
  )
}
