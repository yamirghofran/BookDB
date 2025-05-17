export interface Person {
  id: string; // Assuming Go backend provides ID as string (e.g., UUID)
  name: string;
  email: string;
  age?: number; // Keep for interface flexibility, but not used in current Go backend for people
}

const API_BASE_URL = "/api";

export async function fetchPeople(): Promise<Person[]> {
  const response = await fetch(`${API_BASE_URL}/people`);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ message: 'Failed to fetch people' }));
    throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
  }
  return response.json();
}

export interface AddPersonPayload {
  name: string;
  email: string;
  // age?: number; // Backend CreatePerson expects name and email
}

export async function addPerson(personData: AddPersonPayload): Promise<Person> {
  const response = await fetch(`${API_BASE_URL}/people`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(personData),
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ message: 'Failed to add person' }));
    throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
  }
  return response.json();
}

import type { Book as BookFromTypes, PaginatedUsersResponse } from "./types";

// Interface representing the raw book structure from the backend API
interface RawBookFromAPI {
  ID: string;
  Title: string;
  Authors?: string[]; // Now expecting an array of strings for authors
  CoverImageUrl?: string | null;
  Description?: string | null;
  GoodreadsID?: number;
  GoodreadsUrl?: string | null;
  PublicationYear?: number | null;
  AverageRating?: number | null;
  RatingsCount?: number | null;
  // Include any other fields the backend /api/books actually returns
}

interface FetchBooksParams {
  limit?: number;
  offset?: number;
}

export async function fetchBooks(params?: FetchBooksParams): Promise<BookFromTypes[]> {
  const path = `${API_BASE_URL}/books`;
  const fullUrl = new URL(path, window.location.origin); // Use window.location.origin as base
  if (params?.limit !== undefined) {
    fullUrl.searchParams.append('limit', String(params.limit));
  }
  if (params?.offset !== undefined) {
    fullUrl.searchParams.append('offset', String(params.offset));
  }
  console.log("Constructed URL for fetchBooks:", fullUrl.toString()); // Log the constructed URL

  const response = await fetch(fullUrl.toString());
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ message: 'Failed to fetch books' }));
    throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
  }
  
  const rawBooks: RawBookFromAPI[] = await response.json();
  console.log(`Raw books from API (/books with params ${JSON.stringify(params)}):`, rawBooks);
  
  const mappedBooks = rawBooks.map((rawBook): BookFromTypes => {
    return {
      id: rawBook.ID,
      title: rawBook.Title,
      authors: Array.isArray(rawBook.Authors) ? rawBook.Authors : [],
      coverUrl: rawBook.CoverImageUrl || "",
      description: rawBook.Description === null || rawBook.Description === undefined
                   ? undefined
                   : String(rawBook.Description),
    };
  });
  console.log(`Mapped books for frontend (/books with params ${JSON.stringify(params)}):`, mappedBooks);
  return mappedBooks;
}

interface SearchBooksAPIParams {
  query: string;
  limit?: number;
  offset?: number;
}

interface SearchBooksAPIResponse {
  query: string;
  results: RawBookFromAPI[];
}

export async function searchBooksAPI(params: SearchBooksAPIParams): Promise<BookFromTypes[]> {
  const path = `${API_BASE_URL}/books/search`;
  const fullUrl = new URL(path, window.location.origin); // Use window.location.origin as base
  fullUrl.searchParams.append('q', params.query);
  if (params?.limit !== undefined) {
    fullUrl.searchParams.append('limit', String(params.limit));
  }
  if (params?.offset !== undefined) {
    fullUrl.searchParams.append('offset', String(params.offset));
  }
  console.log("Constructed URL for searchBooksAPI:", fullUrl.toString()); // Log the constructed URL

  const response = await fetch(fullUrl.toString());
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ message: `Failed to search books for query: ${params.query}` }));
    throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
  }

  const searchData: SearchBooksAPIResponse = await response.json();
  console.log(`Raw search results from API (/books/search for query "${params.query}"):`, searchData);

  const mappedBooks = searchData.results.map((rawBook): BookFromTypes => {
    return {
      id: rawBook.ID,
      title: rawBook.Title,
      authors: Array.isArray(rawBook.Authors) ? rawBook.Authors : [],
      coverUrl: rawBook.CoverImageUrl || "",
      description: rawBook.Description === null || rawBook.Description === undefined
                   ? undefined
                   : String(rawBook.Description),
    };
  });
  console.log(`Mapped search results for frontend (query "${params.query}"):`, mappedBooks);
  return mappedBooks;
}

// Interface representing the raw review structure from the backend API
interface RawReviewFromAPI {
  id: string; // Assuming UUIDs are strings
  bookId: string;
  userId: string;
  userName: string;
  userAvatarUrl: string; // Will be empty from backend for now
  rating: number; // Assuming pgtype.Int2 maps to number
  reviewText: string;
  createdAt: string; // Assuming Timestamptz maps to ISO string
  updatedAt: string;
}

// Interface representing the raw book structure from the backend API for a single book
interface RawBookFromAPIById {
  ID: string;
  Title: string;
  Authors?: string[];
  CoverImageUrl?: string | null;
  Description?: string | null;
  GoodreadsID?: number;
  GoodreadsUrl?: string | null;
  PublicationYear?: number | null;
  AverageRating?: number | null;
  RatingsCount?: number | null;
  Genres?: string[]; // Added Genres
  Reviews?: RawReviewFromAPI[]; // Added Reviews
  ReviewsTotalCount?: number; // Added for pagination
  // Genre is not directly part of GetBookByIDRow from backend
}

export interface FetchBookByIdParams {
  reviewsLimit?: number;
  reviewsOffset?: number;
}

export async function fetchBookById(bookId: string, params?: FetchBookByIdParams): Promise<BookFromTypes> {
  const path = `${API_BASE_URL}/books/${bookId}`;
  const fullUrl = new URL(path, window.location.origin);
  if (params?.reviewsLimit !== undefined) {
    fullUrl.searchParams.append('reviewsLimit', String(params.reviewsLimit));
  }
  if (params?.reviewsOffset !== undefined) {
    fullUrl.searchParams.append('reviewsOffset', String(params.reviewsOffset));
  }
  console.log("Constructed URL for fetchBookById:", fullUrl.toString());

  const response = await fetch(fullUrl.toString());
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ message: `Failed to fetch book with id ${bookId}` }));
    throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
  }
  const rawBook: RawBookFromAPIById = await response.json();
  console.log(`Raw book from API (ID: ${bookId}):`, rawBook);

  const mappedReviews = (rawBook.Reviews || []).map((rawReview): import("./types").Review => ({ // Use Review type directly
    id: rawReview.id,
    userId: rawReview.userId,
    userName: rawReview.userName,
    userAvatar: rawReview.userAvatarUrl || "/placeholder.svg?height=50&width=50", // Use placeholder if empty
    rating: rawReview.rating,
    text: rawReview.reviewText,
    date: new Date(rawReview.createdAt).toISOString().split("T")[0], // Format date
  }));

  return {
    id: rawBook.ID,
    title: rawBook.Title,
    authors: Array.isArray(rawBook.Authors) ? rawBook.Authors : [],
    coverUrl: rawBook.CoverImageUrl || "",
    description: rawBook.Description === null || rawBook.Description === undefined
                 ? undefined
                 : String(rawBook.Description),
    averageRating: rawBook.AverageRating === null || rawBook.AverageRating === undefined
                   ? undefined
                   : rawBook.AverageRating, // Map averageRating
    ratingsCount: rawBook.RatingsCount === null || rawBook.RatingsCount === undefined
                  ? undefined
                  : rawBook.RatingsCount, // Map ratingsCount
    genre: Array.isArray(rawBook.Genres) ? rawBook.Genres : [], // Map genres
    reviews: mappedReviews, // Map reviews
    reviewsTotalCount: rawBook.ReviewsTotalCount, // Map total reviews count
  };
}

// RawSimilarBookFromAPI is an alias for RawBookFromAPI, so it now also expects Authors.
type RawSimilarBookFromAPI = RawBookFromAPI;

export async function fetchSimilarBooks(bookId: string): Promise<BookFromTypes[]> {
  const response = await fetch(`${API_BASE_URL}/recommendations/books/${bookId}/similar`);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ message: `Failed to fetch similar books for id ${bookId}` }));
    throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
  }
  const rawSimilarBooks: RawSimilarBookFromAPI[] = await response.json();
  console.log(`Raw similar books from API (for ID: ${bookId}):`, rawSimilarBooks);

  return rawSimilarBooks.map((rawBook): BookFromTypes => ({
    id: rawBook.ID,
    title: rawBook.Title,
    authors: Array.isArray(rawBook.Authors) ? rawBook.Authors : [], // Use authors from API, default to empty array
    coverUrl: rawBook.CoverImageUrl || "",
    description: rawBook.Description === null || rawBook.Description === undefined
                 ? undefined
                 : String(rawBook.Description),
    genre: [], // Defaulting
  }));
}

export interface FetchUsersWithBookParams {
  limit?: number;
  offset?: number;
}

export async function fetchUsersWithBookInLibrary(
  bookId: string,
  params?: FetchUsersWithBookParams,
): Promise<PaginatedUsersResponse> {
  const path = `${API_BASE_URL}/books/${bookId}/library-users`; // Updated path
  const fullUrl = new URL(path, window.location.origin);
  if (params?.limit !== undefined) {
    fullUrl.searchParams.append("limit", String(params.limit));
  }
  if (params?.offset !== undefined) {
    fullUrl.searchParams.append("offset", String(params.offset));
  }
  console.log("Constructed URL for fetchUsersWithBookInLibrary:", fullUrl.toString());

  const response = await fetch(fullUrl.toString());
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({
      message: `Failed to fetch users for book id ${bookId}`,
    }));
    throw new Error(
      errorData.message || `HTTP error! status: ${response.status}`,
    );
  }
  const data: PaginatedUsersResponse = await response.json();
  console.log(
    `Users with book ID ${bookId} from API (params ${JSON.stringify(params)}):`,
    data,
  );
  return data;
}

// You can add more API functions here, for example:
// export async function fetchPersonById(id: string): Promise<Person> { ... }
// export async function updatePerson(id: string, updates: Partial<Person>): Promise<Person> { ... }
// export async function deletePerson(id: string): Promise<void> { ... }