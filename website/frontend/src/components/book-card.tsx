import { Heart } from "lucide-react";
import type { Book } from "@/lib/types";
import { useUserLibrary } from "@/contexts/UserLibraryContext";

interface BookCardProps {
  book: Book;
}

export default function BookCard({ book }: BookCardProps) {
  const { isBookLiked, addBookToLibrary, removeBookFromLibrary } = useUserLibrary();
  const liked = isBookLiked(book.id);

  const handleLikeToggle = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (liked) {
      removeBookFromLibrary(book.id);
    } else {
      addBookToLibrary(book.id);
    }
  };

  return (
    <div className="bg-card rounded-lg overflow-hidden border shadow-sm hover:shadow-md transition-shadow duration-200 h-full">
      <div className="relative pt-[140%]">
        <img
          src={book.coverUrl || "/placeholder.svg"}
          alt={`Cover of ${book.title}`}
          className="absolute inset-0 w-full h-full object-cover"
        />
      </div>

      <div className="p-4">
        <h3 className="font-semibold text-lg line-clamp-1">{book.title}</h3>
        <p className="text-muted-foreground text-sm mb-3 line-clamp-1">
          {book.authors.join(", ")}
        </p>

        <button
          onClick={handleLikeToggle}
          className="flex items-center gap-1 text-sm font-medium"
          aria-pressed={liked}
        >
          <Heart
            className={`h-5 w-5 transition-colors ${
              liked ? "fill-red-500 text-red-500" : "text-muted-foreground"
            }`}
          />
          <span>{liked ? "Liked" : "Like"}</span>
        </button>
      </div>
    </div>
  );
}
