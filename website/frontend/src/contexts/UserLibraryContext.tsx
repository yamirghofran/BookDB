import React, { createContext, useContext, useState, useEffect, useMemo, useCallback } from "react";
import {
  getAnonymousUserId,
  getLibrary,
  addToLibrary as addToLocalStorageLibrary,
  removeFromLibrary as removeFromLocalStorageLibrary,
  isBookInLibrary as isBookInLocalStorageLibrary,
} from "@/lib/localStorageManager";

interface UserLibraryContextType {
  anonymousUserId: string | null;
  libraryBookIds: string[];
  addBookToLibrary: (bookId: string) => void;
  removeBookFromLibrary: (bookId: string) => void;
  isBookLiked: (bookId: string) => boolean;
}

const UserLibraryContext = createContext<UserLibraryContextType | undefined>(undefined);

export const UserLibraryProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [anonymousUserId, setAnonymousUserId] = useState<string | null>(null);
  const [libraryBookIds, setLibraryBookIds] = useState<string[]>([]);

  useEffect(() => {
    const userId = getAnonymousUserId();
    setAnonymousUserId(userId);
    if (userId) {
      setLibraryBookIds(getLibrary(userId));
    }
  }, []);

  const addBookToLibrary = useCallback((bookId: string) => {
    if (anonymousUserId) {
      addToLocalStorageLibrary(anonymousUserId, bookId);
      setLibraryBookIds((prevLibrary) => {
        if (!prevLibrary.includes(bookId)) {
          return [...prevLibrary, bookId];
        }
        return prevLibrary;
      });
    }
  }, [anonymousUserId]);

  const removeBookFromLibrary = useCallback((bookId: string) => {
    if (anonymousUserId) {
      removeFromLocalStorageLibrary(anonymousUserId, bookId);
      setLibraryBookIds((prevLibrary) => prevLibrary.filter((id) => id !== bookId));
    }
  }, [anonymousUserId]);

  const isBookLiked = useCallback((bookId: string): boolean => {
    if (anonymousUserId) {
      return isBookInLocalStorageLibrary(anonymousUserId, bookId);
    }
    return false;
  }, [anonymousUserId, libraryBookIds]); // libraryBookIds dependency ensures re-check when it changes

  // Update localStorage when libraryBookIds state changes (e.g. after initial load or modification)
  // This might be slightly redundant if all modifications also call localStorage directly,
  // but ensures consistency if state were to be modified by other means in the future.
  // However, the current implementation of addBookToLibrary and removeBookFromLibrary already updates localStorage.
  // For now, direct localStorage updates in those functions are sufficient.
  // We rely on isBookLiked to re-evaluate based on localStorage or the state which is synced from it.

  const contextValue = useMemo(() => ({
    anonymousUserId,
    libraryBookIds,
    addBookToLibrary,
    removeBookFromLibrary,
    isBookLiked,
  }), [anonymousUserId, libraryBookIds, addBookToLibrary, removeBookFromLibrary, isBookLiked]);

  return (
    <UserLibraryContext.Provider value={contextValue}>
      {children}
    </UserLibraryContext.Provider>
  );
};

export const useUserLibrary = (): UserLibraryContextType => {
  const context = useContext(UserLibraryContext);
  if (context === undefined) {
    throw new Error("useUserLibrary must be used within a UserLibraryProvider");
  }
  return context;
};