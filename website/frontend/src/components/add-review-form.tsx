import { useState } from "react"
import { X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogClose } from "@/components/ui/dialog"
import type { Review } from "@/lib/types"

interface AddReviewFormProps {
  bookId: string
  isOpen: boolean
  onClose: () => void
  onSubmit: (review: Omit<Review, "id" | "userId" | "date" | "rating" | "userAvatar" | "userName">) => void
}

export default function AddReviewForm({ bookId, isOpen, onClose, onSubmit }: AddReviewFormProps) {
  const [reviewText, setReviewText] = useState("")
  const [errors, setErrors] = useState<{ text?: string; name?: string }>({})

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    // Validate form
    const newErrors: { text?: string; name?: string } = {}

    if (!reviewText.trim()) {
      newErrors.text = "Please enter your review"
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      return
    }

    // Submit the review
    onSubmit({
      text: reviewText,
    })

    // Reset form
    setReviewText("")
    setErrors({})
    onClose()
  }

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Write a Review</DialogTitle>
          <DialogClose className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-accent data-[state=open]:text-muted-foreground">
            <X className="h-4 w-4" />
            <span className="sr-only">Close</span>
          </DialogClose>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Textarea
              id="review"
              value={reviewText}
              onChange={(e) => setReviewText(e.target.value)}
              placeholder="Share your thoughts about this book..."
              rows={4}
            />
            {errors.text && <p className="text-red-500 text-xs mt-1">{errors.text}</p>}
          </div>

          <div className="flex justify-end gap-2">
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit">Submit Review</Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}
