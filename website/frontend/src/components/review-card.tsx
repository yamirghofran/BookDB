import type { Review } from "@/lib/types"

interface ReviewCardProps {
  review: Review
}

export default function ReviewCard({ review }: ReviewCardProps) {
  return (
    <div className="border rounded-lg p-4 mb-4">
      <div className="flex flex-col">
        {/* Review content */}
        <h4 className="font-medium mb-2">{review.userName}</h4>
        <p className="text-sm text-muted-foreground">{review.text}</p>
      </div>
    </div>
  )
}
