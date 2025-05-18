import type { Person, Book as BookFromTypes, PaginatedUsersResponse, PersonDetails } from "./types"; // Added Person to import

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
  // For recommendations, the backend now returns the full BookResponse structure
  Genres?: string[];
  // Reviews and ReviewsTotalCount are not expected in lists like recommendations
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
      genre: Array.isArray(rawBook.Genres) ? rawBook.Genres : [], // Map genres if available
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
      genre: Array.isArray(rawBook.Genres) ? rawBook.Genres : [], // Map genres if available
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
type RawSimilarBookFromAPI = RawBookFromAPI; // This should be fine as RawBookFromAPI now includes Genres

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
    genre: Array.isArray(rawBook.Genres) ? rawBook.Genres : [], // Map genres if available
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

export async function fetchAnonymousRecommendations(likedBookIds: string[]): Promise<BookFromTypes[]> {
  const response = await fetch(`${API_BASE_URL}/recommendations/anonymous`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ likedBookIds }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ message: 'Failed to fetch anonymous recommendations' }));
    throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
  }

  const rawRecommendedBooks: RawBookFromAPI[] = await response.json(); // Backend returns full BookResponse
  console.log(`Raw anonymous recommendations from API (likedBookIds: ${likedBookIds.join(', ')}):`, rawRecommendedBooks);

  // The backend already returns BookResponse which is compatible with RawBookFromAPI here
  // (assuming BookResponse includes ID, Title, Authors, CoverImageUrl, Description, Genres)
  return rawRecommendedBooks.map((rawBook): BookFromTypes => ({
    id: rawBook.ID,
    title: rawBook.Title,
    authors: Array.isArray(rawBook.Authors) ? rawBook.Authors : [],
    coverUrl: rawBook.CoverImageUrl || "",
    description: rawBook.Description === null || rawBook.Description === undefined
                 ? undefined
                 : String(rawBook.Description),
    genre: Array.isArray(rawBook.Genres) ? rawBook.Genres : [],
    // Other fields like averageRating, ratingsCount might be present if backend sends them
    // and BookFromTypes includes them. For now, mapping the core fields.
  }));
}

// Raw type definitions for data coming from the backend for person details
interface RawUserLibraryBookDetail { // Keep this internal to api.ts if only used for mapping here
  ID: string;
  Title: string;
  CoverImageUrl?: {
    String: string;
    Valid: boolean;
  } | string | null;
  Authors: string[];
}

interface RawUserReviewWithBookInfo { // Keep this internal to api.ts if only used for mapping here
  reviewId: string;
  bookId: string;
  bookTitle: string;
  bookCoverImageUrl?: { // Corrected to lowercase 'b' to match JSON tag
    String: string;
    Valid: boolean;
  } | string | null;
  userId: string;
  rating: { Int16: number, Valid: boolean } | number | null;
  reviewText: string;
  reviewCreatedAt: { Time: string, Valid: boolean } | string;
  reviewUpdatedAt: { Time: string, Valid: boolean } | string;
}

export async function fetchPersonDetails(userId: string): Promise<PersonDetails> {
  const response = await fetch(`${API_BASE_URL}/people/${userId}/details`);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ message: `Failed to fetch details for user ${userId}` }));
    throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
  }
  const rawData = await response.json();
  console.log(`Raw person details from API (user ID: ${userId}):`, rawData);

  // Map rawData.user (db.User) to frontend Person type
  const mappedUser: Person = {
    id: rawData.user.ID, // Corrected to capitalized ID
    name: rawData.user.Name, // Corrected to capitalized Name
    email: rawData.user.Email, // Corrected to capitalized Email
    // age is not in db.User
  };

  // Map rawData.libraryBooks to frontend BookFromTypes[]
  const mappedLibraryBooks: BookFromTypes[] = (rawData.libraryBooks || []).map((rawBook: RawUserLibraryBookDetail): BookFromTypes => {
    let coverUrl = "";
    if (rawBook.CoverImageUrl) {
      if (typeof rawBook.CoverImageUrl === 'string') {
        coverUrl = rawBook.CoverImageUrl;
      } else if (rawBook.CoverImageUrl.Valid) {
        coverUrl = rawBook.CoverImageUrl.String;
      }
    }
    return {
      id: rawBook.ID,
      title: rawBook.Title,
      authors: Array.isArray(rawBook.Authors) ? rawBook.Authors : [],
      coverUrl: coverUrl,
      // description, genre, etc., are not in RawUserLibraryBookDetail, so they'll be undefined
    };
  });

  // Map rawData.userReviews to frontend Review[]
  const mappedUserReviews: import("./types").Review[] = (rawData.userReviews || []).map((rawReview: RawUserReviewWithBookInfo): import("./types").Review => {
    let rating = 0;
    if (rawReview.rating) {
      if (typeof rawReview.rating === 'number') {
        rating = rawReview.rating;
      } else if (rawReview.rating.Valid) {
        rating = rawReview.rating.Int16;
      }
    }
    let createdAt = "";
     if (rawReview.reviewCreatedAt) {
      if (typeof rawReview.reviewCreatedAt === 'string') {
        createdAt = new Date(rawReview.reviewCreatedAt).toISOString().split("T")[0];
      } else if (rawReview.reviewCreatedAt.Valid) {
        createdAt = new Date(rawReview.reviewCreatedAt.Time).toISOString().split("T")[0];
      }
    }
    let bookCoverUrl = "";
    // Ensure rawReview.bookCoverImageUrl is accessed correctly (lowercase 'b')
    if (rawReview.bookCoverImageUrl) {
      if (typeof rawReview.bookCoverImageUrl === 'string') {
        bookCoverUrl = rawReview.bookCoverImageUrl;
      } else if (rawReview.bookCoverImageUrl.Valid) { // Check Valid for pgtype.Text like objects
        bookCoverUrl = rawReview.bookCoverImageUrl.String;
      }
    }

    return {
      id: rawReview.reviewId,
      userId: rawReview.userId,
      bookId: rawReview.bookId,
      bookTitle: rawReview.bookTitle,
      bookCoverUrl: bookCoverUrl,
      // userName for ReviewCard should be the person whose page we are on.
      // The backend sends rawReview.userId, but ReviewCard expects userName.
      // We can use mappedUser.name here.
      userName: mappedUser.name,
      userAvatar: "/placeholder.svg?height=50&width=50", // Placeholder, not in RawUserReviewWithBookInfo
      rating: rating,
      text: rawReview.reviewText,
      date: createdAt,
    };
  });

  return {
    user: mappedUser,
    libraryBooks: mappedLibraryBooks,
    userReviews: mappedUserReviews,
  };
}

export async function fetchSimilarUsers(userId: string): Promise<Person[]> {
  const response = await fetch(`${API_BASE_URL}/recommendations/users/${userId}/similar`);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ message: `Failed to fetch similar users for user ID ${userId}` }));
    throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
  }
  const rawUsers: any[] = await response.json(); // Backend returns []db.User
  console.log(`Raw similar users from API (for user ID: ${userId}):`, rawUsers);

  // Map []db.User to frontend Person[]
  return rawUsers.map((rawUser: any): Person => ({
    id: rawUser.ID, // Assuming backend sends ID, Name, Email capitalized
    name: rawUser.Name,
    email: rawUser.Email,
    // age is not part of db.User
  }));
}

export async function fetchBookRecommendationsForUser(userId: string): Promise<BookFromTypes[]> {
  const response = await fetch(`${API_BASE_URL}/recommendations/users/${userId}/books`);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ message: `Failed to fetch book recommendations for user ID ${userId}` }));
    throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
  }
  const rawBooks: RawBookFromAPI[] = await response.json(); // Backend returns []BookResponse
  console.log(`Raw book recommendations for user from API (user ID: ${userId}):`, rawBooks);

  return rawBooks.map((rawBook): BookFromTypes => ({
    id: rawBook.ID,
    title: rawBook.Title,
    authors: Array.isArray(rawBook.Authors) ? rawBook.Authors : [],
    coverUrl: rawBook.CoverImageUrl || "",
    description: rawBook.Description === null || rawBook.Description === undefined
                 ? undefined
                 : String(rawBook.Description),
    genre: Array.isArray(rawBook.Genres) ? rawBook.Genres : [],
  }));
}


// You can add more API functions here, for example:
// export async function fetchPersonById(id: string): Promise<Person> { ... }
// export async function updatePerson(id: string, updates: Partial<Person>): Promise<Person> { ... }
// export async function deletePerson(id: string): Promise<void> { ... }