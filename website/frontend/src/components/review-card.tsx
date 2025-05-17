import type { Review } from "@/lib/types"
import { useState } from "react"

interface ReviewCardProps {
  review: Review
}

const MAX_WORDS = 60

export default function ReviewCard({ review }: ReviewCardProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  const words = review.text.split(/\s+/)
  const isTruncatable = words.length > MAX_WORDS

  const displayText =
    isTruncatable && !isExpanded
      ? words.slice(0, MAX_WORDS).join(" ") + "..."
      : review.text

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded)
  }

  return (
    <div className="border rounded-lg p-4 mb-4">
      <div className="flex flex-col">
        {/* Review content */}
        <h4 className="font-medium mb-2">{review.userName}</h4>
        <p className="text-sm text-muted-foreground whitespace-pre-wrap">{displayText}</p>
        {isTruncatable && (
          <button
            onClick={toggleExpanded}
            className="text-blue-600 hover:text-blue-800 text-sm self-start mt-1"
          >
            {isExpanded ? "less..." : "more..."}
          </button>
        )}
      </div>
    </div>
  )
}
