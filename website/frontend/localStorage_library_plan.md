# Plan: Anonymous User Library with localStorage

This document outlines the plan to implement a client-side "liked books" library feature using `localStorage` and an anonymous UUID for users who are not signed in.

## 1. Anonymous User & Library Management (`localStorage` Utilities)

*   **File:** `website/frontend/src/lib/localStorageManager.ts` (New file)
*   **Anonymous User ID:**
    *   `getAnonymousUserId()`:
        *   Checks `localStorage` for an existing `anonymousUserId`.
        *   If not found, generates a new UUID (using `crypto.randomUUID()`).
        *   Stores the new UUID in `localStorage`.
        *   Returns the UUID.
*   **Library Management:**
    *   `LIBRARY_STORAGE_KEY_PREFIX = "userLibrary_"`
    *   `getLibraryStorageKey(userId: string): string`: Returns `LIBRARY_STORAGE_KEY_PREFIX + userId`.
    *   `getLibrary(userId: string): string[]`:
        *   Retrieves the list of book IDs from `localStorage` using `getLibraryStorageKey(userId)`.
        *   Parses the JSON string (or returns an empty array if not found/invalid).
    *   `addToLibrary(userId: string, bookId: string): void`:
        *   Gets the current library.
        *   Adds `bookId` if not already present.
        *   Saves the updated library to `localStorage`.
    *   `removeFromLibrary(userId: string, bookId: string): void`:
        *   Gets the current library.
        *   Removes `bookId`.
        *   Saves the updated library to `localStorage`.
    *   `isBookInLibrary(userId: string, bookId: string): boolean`:
        *   Checks if `bookId` exists in the user's library.

## 2. React Context for User and Library State

*   **File:** `website/frontend/src/contexts/UserLibraryContext.tsx` (New file)
*   **Context Definition:**
    *   `UserLibraryContext`:
        *   `anonymousUserId: string | null`
        *   `libraryBookIds: string[]`
        *   `addBookToLibrary: (bookId: string) => void`
        *   `removeBookFromLibrary: (bookId: string) => void`
        *   `isBookLiked: (bookId: string) => boolean`
*   **Provider Component:**
    *   `UserLibraryProvider`:
        *   On mount, calls `getAnonymousUserId()` from `localStorageManager.ts` to initialize `anonymousUserId`.
        *   If `anonymousUserId` is available, calls `getLibrary()` to initialize `libraryBookIds`.
        *   Provides the state and memoized versions of the management functions.
        *   Uses `useEffect` to update `localStorage` whenever `libraryBookIds` changes.
*   **Hook:**
    *   `useUserLibrary(): UserLibraryContext`: Custom hook to easily consume the context.

## 3. Integrating the Context Provider

*   **File:** `website/frontend/src/routes/__root.tsx`
*   Wrap the main layout content with `UserLibraryProvider`.

    ```tsx
    // In website/frontend/src/routes/__root.tsx
    import { UserLibraryProvider } from "@/contexts/UserLibraryContext"; // Adjust path

    export const Route = createRootRouteWithContext<MyRouterContext>()({
      component: () => (
        <UserLibraryProvider> {/* Wrap here */}
          <>
            <Navbar />
            <Outlet />
            <TanStackRouterDevtools />
            <TanstackQueryLayout />
          </>
        </UserLibraryProvider>
      ),
    });
    ```

## 4. Updating UI Components

*   **Component:** `website/frontend/src/components/book-card.tsx`
    *   Remove the local `isLiked` state.
    *   Use the `useUserLibrary` hook to get `isBookLiked`, `addBookToLibrary`, and `removeBookFromLibrary`.
    *   The "Like" button's `onClick` handler will now call `addBookToLibrary` or `removeBookFromLibrary` based on the current liked status.
    *   The heart icon and text will be determined by `isBookLiked(book.id)`.

*   **New Component (Optional):** `website/frontend/src/routes/library.tsx`
    *   A new route to display all books in the user's library.
    *   Uses `useUserLibrary` to get `libraryBookIds`.
    *   Fetches book details for these IDs.

## 5. UUID Generation

*   Utilize the browser-native `crypto.randomUUID()` for generating UUIDs, which avoids needing an external package for this specific task.

## Visual Plan (Mermaid Diagram)

```mermaid
graph TD
    A[User Action: Likes a Book] --> B{BookCard Component};
    B --> C{useUserLibrary Hook};
    C --> D[UserLibraryContext];
    D --> E{UserLibraryProvider};
    E --> F{localStorageManager Utilities};
    F --> G[localStorage: anonymousUserId];
    F --> H[localStorage: userLibrary_{userId}];

    subgraph Frontend Application
        direction LR
        AA[main.tsx] --> BB[__root.tsx];
        BB -- Wraps with --> E;
        E -- Provides --> D;
        B -- Consumes --> C;
    end

    subgraph Local Storage Management
        direction TB
        F -- Manages --> G;
        F -- Manages --> H;
    end

    %% Styling
    classDef component fill:#f9f,stroke:#333,stroke-width:2px;
    classDef hook fill:#ccf,stroke:#333,stroke-width:2px;
    classDef context fill:#cfc,stroke:#333,stroke-width:2px;
    classDef provider fill:#ffc,stroke:#333,stroke-width:2px;
    classDef utility fill:#fcf,stroke:#333,stroke-width:2px;
    classDef storage fill:#eee,stroke:#333,stroke-width:2px;

    class B,AA,BB component;
    class C hook;
    class D context;
    class E provider;
    class F utility;
    class G,H storage;