import type { Book } from "./types"

export const books: Book[] = [
  {
    id: "1",
    title: "The Midnight Library",
    authors: ["Matt Haig"],
    coverUrl: "/placeholder.svg?height=400&width=300",
    description:
      "Between life and death there is a library, and within that library, the shelves go on forever. Every book provides a chance to try another life you could have lived. To see how things would be if you had made other choices... Would you have done anything different, if you had the chance to undo your regrets?",
    genre: ["Fiction", "Fantasy", "Contemporary"],
    reviews: [
      {
        id: "r1",
        userId: "u1",
        userName: "Alex Johnson",
        userAvatar: "/placeholder.svg?height=50&width=50",
        rating: 5,
        text: "This book changed my perspective on life. The concept is brilliant and the execution is flawless. I couldn't put it down!",
        date: "2023-10-15",
      },
      {
        id: "r2",
        userId: "u2",
        userName: "Sarah Miller",
        userAvatar: "/placeholder.svg?height=50&width=50",
        rating: 4,
        text: "A beautiful exploration of regret and second chances. The writing is poetic and the story is deeply moving.",
        date: "2023-09-22",
      },
      {
        id: "r3",
        userId: "u3",
        userName: "Michael Chen",
        userAvatar: "/placeholder.svg?height=50&width=50",
        rating: 5,
        text: "One of the most thought-provoking books I've read this year. It makes you reflect on your own choices and the infinite possibilities of life.",
        date: "2023-11-05",
      },
    ],
  },
  {
    id: "2",
    title: "Atomic Habits",
    authors: ["James Clear"],
    coverUrl: "/placeholder.svg?height=400&width=300",
    description:
      "An Easy & Proven Way to Build Good Habits & Break Bad Ones. No matter your goals, Atomic Habits offers a proven framework for improving every day. James Clear, one of the world's leading experts on habit formation, reveals practical strategies that will teach you exactly how to form good habits, break bad ones, and master the tiny behaviors that lead to remarkable results.",
    genre: ["Self-help", "Productivity", "Psychology"],
    reviews: [
      {
        id: "r4",
        userId: "u4",
        userName: "Emily Roberts",
        userAvatar: "/placeholder.svg?height=50&width=50",
        rating: 5,
        text: "This book has completely transformed my approach to building habits. The 1% better every day concept is simple yet powerful.",
        date: "2023-08-30",
      },
      {
        id: "r5",
        userId: "u5",
        userName: "David Wong",
        userAvatar: "/placeholder.svg?height=50&width=50",
        rating: 4,
        text: "Practical, actionable advice backed by science. I've already implemented several of the strategies with great results.",
        date: "2023-10-12",
      },
    ],
  },
  {
    id: "3",
    title: "Educated",
    authors: ["Tara Westover"],
    coverUrl: "/placeholder.svg?height=400&width=300",
    description:
      "A Memoir of a journey from no formal education to a PhD at Cambridge University. Born to survivalists in the mountains of Idaho, Tara Westover was seventeen the first time she set foot in a classroom. Her family was so isolated from mainstream society that there was no one to ensure the children received an education, and no one to intervene when one of Tara's older brothers became violent.",
    genre: ["Memoir", "Biography", "Autobiography"],
    reviews: [
      {
        id: "r6",
        userId: "u6",
        userName: "Jennifer Adams",
        userAvatar: "/placeholder.svg?height=50&width=50",
        rating: 5,
        text: "An incredible story of resilience and the power of education. Tara's journey is both heartbreaking and inspiring.",
        date: "2023-07-18",
      },
    ],
  },
  {
    id: "4",
    title: "The Silent Patient",
    authors: ["Alex Michaelides"],
    coverUrl: "/placeholder.svg?height=400&width=300",
    description:
      "A psychological thriller about a woman's act of violence against her husband. Alicia Berenson's life is seemingly perfect. A famous painter married to an in-demand fashion photographer, she lives in a grand house with big windows overlooking a park in one of London's most desirable areas. One evening her husband Gabriel returns home late from a fashion shoot, and Alicia shoots him five times in the face, and then never speaks another word.",
    genre: ["Thriller", "Mystery", "Psychological Fiction"],
    reviews: [
      {
        id: "r7",
        userId: "u7",
        userName: "Robert Thompson",
        userAvatar: "/placeholder.svg?height=50&width=50",
        rating: 4,
        text: "A gripping thriller with an unexpected twist. The ending completely caught me off guard!",
        date: "2023-09-05",
      },
      {
        id: "r8",
        userId: "u8",
        userName: "Lisa Garcia",
        userAvatar: "/placeholder.svg?height=50&width=50",
        rating: 5,
        text: "One of the best psychological thrillers I've read. The author masterfully builds tension throughout the story.",
        date: "2023-10-20",
      },
    ],
  },
  {
    id: "5",
    title: "Where the Crawdads Sing",
    authors: ["Delia Owens"],
    coverUrl: "/placeholder.svg?height=400&width=300",
    description:
      "A novel about a young woman who raised herself in the marshes of the deep South. For years, rumors of the 'Marsh Girl' have haunted Barkley Cove, a quiet town on the North Carolina coast. So in late 1969, when handsome Chase Andrews is found dead, the locals immediately suspect Kya Clark, the so-called Marsh Girl.",
    genre: ["Fiction", "Mystery", "Literary Fiction"],
    reviews: [],
  },
  {
    id: "6",
    title: "Becoming",
    authors: ["Michelle Obama"],
    coverUrl: "/placeholder.svg?height=400&width=300",
    description:
      "The memoir of the former First Lady of the United States. In a life filled with meaning and accomplishment, Michelle Obama has emerged as one of the most iconic and compelling women of our era. As First Lady of the United States of America—the first African American to serve in that role—she helped create the most welcoming and inclusive White House in history.",
    genre: ["Memoir", "Autobiography", "Biography"],
    reviews: [],
  },
  {
    id: "7",
    title: "The Alchemist",
    authors: ["Paulo Coelho"],
    coverUrl: "/placeholder.svg?height=400&width=300",
    description:
      "A fable about following your dream. Paulo Coelho's masterpiece tells the mystical story of Santiago, an Andalusian shepherd boy who yearns to travel in search of a worldly treasure. His quest will lead him to riches far different—and far more satisfying—than he ever imagined.",
    genre: ["Fiction", "Fantasy", "Philosophy"],
    reviews: [],
  },
  {
    id: "8",
    title: "Sapiens: A Brief History of Humankind",
    authors: ["Yuval Noah Harari"],
    coverUrl: "/placeholder.svg?height=400&width=300",
    description:
      "A study of the history of mankind from the evolution of archaic human species. One hundred thousand years ago, at least six different species of humans inhabited Earth. Yet today there is only one—homo sapiens. What happened to the others? And what may happen to us?",
    genre: ["History", "Science", "Anthropology"],
    reviews: [],
  },
  {
    id: "9",
    title: "The Four Winds",
    authors: ["Kristin Hannah"],
    coverUrl: "/placeholder.svg?height=400&width=300",
    description:
      "A story set during the Great Depression and the Dust Bowl. Texas, 1934. Millions are out of work and a drought has broken the Great Plains. Farmers are fighting to keep their land and their livelihoods as the crops are failing, the water is drying up, and dust threatens to bury them all.",
    genre: ["Historical Fiction", "Fiction", "Drama"],
    reviews: [],
  },
  {
    id: "10",
    title: "Project Hail Mary",
    authors: ["Andy Weir"],
    coverUrl: "/placeholder.svg?height=400&width=300",
    description:
      "A lone astronaut must save the earth from disaster. Ryland Grace is the sole survivor on a desperate, last-chance mission—and if he fails, humanity and the earth itself will perish. Except that right now, he doesn't know that. He can't even remember his own name, let alone the nature of his assignment or how to complete it.",
    genre: ["Science Fiction", "Adventure", "Space"],
    reviews: [],
  },
  {
    id: "11",
    title: "The Invisible Life of Addie LaRue",
    authors: ["V.E. Schwab"],
    coverUrl: "/placeholder.svg?height=400&width=300",
    description:
      "A story about a woman who makes a Faustian bargain to live forever. France, 1714: in a moment of desperation, a young woman makes a Faustian bargain to live forever—and is cursed to be forgotten by everyone she meets. Thus begins the extraordinary life of Addie LaRue, and a dazzling adventure that will play out across centuries and continents.",
    genre: ["Fantasy", "Historical Fiction", "Romance"],
    reviews: [],
  },
  {
    id: "12",
    title: "Klara and the Sun",
    authors: ["Kazuo Ishiguro"],
    coverUrl: "/placeholder.svg?height=400&width=300",
    description:
      "A story told from the perspective of an Artificial Friend. From the bestselling author of Never Let Me Go and The Remains of the Day, a stunning new novel that asks, what does it mean to love?",
    genre: ["Science Fiction", "Literary Fiction", "Dystopian"],
    reviews: [],
  },
]
