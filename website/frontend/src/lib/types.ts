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
  }
  