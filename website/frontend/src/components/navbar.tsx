import { useState } from "react"
import { Link } from "@tanstack/react-router"
import { Menu, X, BookOpen } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        {/* Logo */}
        <div className="mx-6">
          <Link to={'/' as any} className="flex items-center gap-2 font-bold text-xl">
            <BookOpen className="h-6 w-6" />
            <span>BookDB</span>
          </Link>
        </div>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center justify-center gap-6">
          <Link
            to={'/for-you' as any}
            className="text-sm font-medium transition-colors hover:text-primary"
            activeProps={{ className: "text-primary font-semibold" }}
            inactiveProps={{ className: "text-muted-foreground" }}
          >
            For You
          </Link>
          <Link
            to={'/library' as any}
            className="text-sm font-medium transition-colors hover:text-primary"
            activeProps={{ className: "text-primary font-semibold" }}
            inactiveProps={{ className: "text-muted-foreground" }}
          >
            Library
          </Link>
          <Link
            to={'/authors' as any}
            className="text-sm font-medium transition-colors hover:text-primary"
            activeProps={{ className: "text-primary font-semibold" }}
            inactiveProps={{ className: "text-muted-foreground" }}
          >
            Authors
          </Link>
        </nav>

        {/* Sign Up/In Button */}
        <div className="hidden md:flex items-center justify-end gap-4">
          <Link to={'/login' as any} preload={false}>
            <Button variant="outline" size="sm">
              Sign up / in
            </Button>
          </Link>
        </div>

        {/* Mobile Menu Button */}
        <button
          className="md:hidden ml-auto"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          aria-label={isMenuOpen ? "Close menu" : "Open menu"}
        >
          {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </button>
      </div>

      {/* Mobile Menu */}
      {isMenuOpen && (
        <div className="md:hidden border-t">
          <div className="container py-4 flex flex-col gap-4">
            <Link
              to={'/for-you' as any}
              className="px-2 py-2 text-sm font-medium rounded-md"
              activeProps={{ className: "bg-muted text-primary font-semibold" }}
              inactiveProps={{ className: "text-muted-foreground" }}
              onClick={() => setIsMenuOpen(false)}
            >
              For You
            </Link>
            <Link
              to={'/library' as any}
              className="px-2 py-2 text-sm font-medium rounded-md"
              activeProps={{ className: "bg-muted text-primary font-semibold" }}
              inactiveProps={{ className: "text-muted-foreground" }}
              onClick={() => setIsMenuOpen(false)}
            >
              Library
            </Link>
            <Link
              to={'/authors' as any}
              className="px-2 py-2 text-sm font-medium rounded-md"
              activeProps={{ className: "bg-muted text-primary font-semibold" }}
              inactiveProps={{ className: "text-muted-foreground" }}
              onClick={() => setIsMenuOpen(false)}
            >
              Authors
            </Link>
            <div className="pt-2 border-t">
              <Link to={'/login' as any} preload={false} className="w-full" onClick={() => setIsMenuOpen(false)}>
                <Button className="w-full" variant="outline" size="sm">
                  Sign up / in
                </Button>
              </Link>
            </div>
          </div>
        </div>
      )}
    </header>
  )
}
