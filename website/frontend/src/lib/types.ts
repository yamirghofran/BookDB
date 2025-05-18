// Add Person interface here
export interface Person {
  id: string;
  name: string;
  email: string;
  age?: number;
}

export interface Book {
    id: string
    title: string
    authors: string[]
    coverUrl: string
    description?: string
    genre?: string[]
    reviews?: Review[]
    averageRating?: number // Added
    ratingsCount?: number  // Added
    reviewsTotalCount?: number // Added for pagination
  }
  
  export interface Author {
    id: string
    name: string
    imageUrl?: string
    birthYear?: number
    deathYear?: number
    bio?: string
  }
  
  export interface Review {
    id: string
    userId: string
    userName: string
    text: string
    date: string
    rating?: number
    userAvatar?: string
    bookId?: string;   // Added for context on PersonDetails page
    bookTitle?: string; // Added for context on PersonDetails page
    bookCoverUrl?: string; // Added for PersonReviewCard
  }

export interface UserInLibrary {
  id: string;
  name: string;
  // avatarUrl?: string; // Add if you include this from the backend
}

export interface PaginatedUsersResponse {
  users: UserInLibrary[];
  totalUsers: number;
  page: number;
  limit: number;
}

export interface PersonDetails {
  user: Person;
  libraryBooks: Book[]; // Uses the existing Book type
  userReviews: Review[]; // Uses the existing Review type
}