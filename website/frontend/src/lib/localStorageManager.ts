const ANONYMOUS_USER_ID_KEY = "anonymousUserId";
const LIBRARY_STORAGE_KEY_PREFIX = "userLibrary_";

/**
 * Retrieves the anonymous user ID from localStorage.
 * If not found, generates a new UUID, stores it, and returns it.
 */
export function getAnonymousUserId(): string {
  let userId = localStorage.getItem(ANONYMOUS_USER_ID_KEY);
  if (!userId) {
    userId = crypto.randomUUID();
    localStorage.setItem(ANONYMOUS_USER_ID_KEY, userId);
  }
  return userId;
}

/**
 * Generates the localStorage key for a user's library.
 * @param userId The ID of the user.
 */
export function getLibraryStorageKey(userId: string): string {
  return `${LIBRARY_STORAGE_KEY_PREFIX}${userId}`;
}

/**
 * Retrieves the list of book IDs from the user's library in localStorage.
 * @param userId The ID of the user.
 */
export function getLibrary(userId: string): string[] {
  const storageKey = getLibraryStorageKey(userId);
  const libraryJson = localStorage.getItem(storageKey);
  if (libraryJson) {
    try {
      const library = JSON.parse(libraryJson);
      if (Array.isArray(library)) {
        return library.filter(id => typeof id === 'string');
      }
    } catch (error) {
      console.error("Error parsing library from localStorage:", error);
      // If parsing fails, treat it as an empty library
      localStorage.removeItem(storageKey); // Clear corrupted data
      return [];
    }
  }
  return [];
}

/**
 * Adds a book to the user's library in localStorage.
 * @param userId The ID of the user.
 * @param bookId The ID of the book to add.
 */
export function addToLibrary(userId: string, bookId: string): void {
  const library = getLibrary(userId);
  if (!library.includes(bookId)) {
    const updatedLibrary = [...library, bookId];
    localStorage.setItem(getLibraryStorageKey(userId), JSON.stringify(updatedLibrary));
  }
}

/**
 * Removes a book from the user's library in localStorage.
 * @param userId The ID of the user.
 * @param bookId The ID of the book to remove.
 */
export function removeFromLibrary(userId: string, bookId: string): void {
  let library = getLibrary(userId);
  if (library.includes(bookId)) {
    library = library.filter((id) => id !== bookId);
    localStorage.setItem(getLibraryStorageKey(userId), JSON.stringify(library));
  }
}

/**
 * Checks if a book is in the user's library.
 * @param userId The ID of the user.
 * @param bookId The ID of the book to check.
 */
export function isBookInLibrary(userId: string, bookId: string): boolean {
  const library = getLibrary(userId);
  return library.includes(bookId);
}