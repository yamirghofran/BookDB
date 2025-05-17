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

import type { Book as BookFromTypes } from "./types";

// Interface representing the raw book structure from the backend API
interface RawBookFromAPI {
  id: string;
  title: string;
  // authors is not directly provided by the /api/books list endpoint currently
  // If it were, it would be: authors?: any[]; (or specific raw author type)
  cover_image_url?: string | null;
  description?: string | null;
  goodreads_id?: number;
  goodreads_url?: string | null;
  publication_year?: number | null;
  average_rating?: number | null;
  ratings_count?: number | null;
  // Include any other fields the backend /api/books actually returns
}

export async function fetchBooks(): Promise<BookFromTypes[]> {
  const response = await fetch(`${API_BASE_URL}/books`);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ message: 'Failed to fetch books' }));
    throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
  }
  
  const rawBooks: RawBookFromAPI[] = await response.json();
  
  return rawBooks.map((rawBook): BookFromTypes => {
    return {
      id: rawBook.id,
      title: rawBook.title,
      authors: [], // Backend /api/books list doesn't provide authors, default to empty array.
                     // If backend provided rawBook.authors, mapping would be:
                     // authors: Array.isArray(rawBook.authors) ? rawBook.authors.map(String) : [],
      coverUrl: rawBook.cover_image_url || "", // Map and provide default
      description: rawBook.description === null || rawBook.description === undefined
                   ? undefined
                   : String(rawBook.description), // Map null to undefined
      // genre and reviews are optional in BookFromTypes and not expected from this endpoint.
    };
  });
}

// You can add more API functions here, for example:
// export async function fetchPersonById(id: string): Promise<Person> { ... }
// export async function updatePerson(id: string, updates: Partial<Person>): Promise<Person> { ... }
// export async function deletePerson(id: string): Promise<void> { ... }