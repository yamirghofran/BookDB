import type { Review } from "@/lib/types";
import { useState } from "react";
import { Link } from "@tanstack/react-router";
import { Star } from "lucide-react"; // For displaying rating

interface PersonReviewCardProps {
  review: Review;
}

const MAX_WORDS_PERSON_REVIEW = 50; // Slightly shorter for this layout potentially

export default function PersonReviewCard({ review }: PersonReviewCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const words = review.text.split(/\s+/);
  const isTruncatable = words.length > MAX_WORDS_PERSON_REVIEW;

  const displayText =
    isTruncatable && !isExpanded
      ? words.slice(0, MAX_WORDS_PERSON_REVIEW).join(" ") + "..."
      : review.text;

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className="bg-card border p-4 rounded-lg shadow flex gap-4">
      {/* Left: Book Cover */}
      <div className="w-1/4 md:w-1/5 flex-shrink-0">
        {review.bookId && review.bookCoverUrl && ( // Ensure bookCoverUrl exists before trying to display
          <Link to="/books/$id" params={{ id: review.bookId }} className="block aspect-[2/3] relative rounded overflow-hidden shadow-sm bg-muted"> {/* Added bg-muted for placeholder visibility */}
            <img
              src={review.bookCoverUrl} // Removed fallback to placeholder here, handled by conditional rendering
              alt={`Cover of ${review.bookTitle || "book"}`}
              className="object-cover w-full h-full"
              onError={(e) => (e.currentTarget.src = "/placeholder.svg?text=No+Cover")} // Fallback for broken image links
            />
          </Link>
        )}
        {/* Fallback if no bookCoverUrl or bookId */}
        {(!review.bookId || !review.bookCoverUrl) && (
           <div className="block aspect-[2/3] relative rounded overflow-hidden shadow-sm bg-muted flex items-center justify-center">
             <span className="text-xs text-muted-foreground">No Cover</span>
           </div>
        )}
      </div>

      {/* Right: Review Details */}
      <div className="flex-grow flex flex-col">
        {review.bookTitle && (
          <h3 className="text-lg font-semibold mb-1 leading-tight">
            <Link to="/books/$id" params={{ id: review.bookId || "" }} className="text-primary hover:underline">{review.bookTitle}</Link> {/* Removed "Review for " */}
          </h3>
        )}
        {review.rating !== undefined && review.rating > 0 && (
          <div className="flex items-center mb-2">
            {Array.from({ length: 5 }, (_, i) => (
              <Star
                key={i}
                className={`h-4 w-4 ${
                  i < review.rating! ? "text-yellow-400 fill-yellow-400" : "text-muted-foreground"
                }`}
              />
            ))}
            <span className="ml-2 text-xs text-muted-foreground">({review.rating}/5)</span>
          </div>
        )}
        <p className="text-sm text-muted-foreground whitespace-pre-wrap flex-grow">{displayText}</p>
        {isTruncatable && (
          <button
            onClick={toggleExpanded}
            className="text-blue-600 hover:text-blue-800 text-sm self-start mt-1"
          >
            {isExpanded ? "less..." : "more..."}
          </button>
        )}
        {/* <p className="text-xs text-muted-foreground mt-2 text-right">Reviewed on: {review.date}</p> */}{/* Removed timestamp */}
      </div>
    </div>
  );
}