import { Outlet, createRootRouteWithContext } from "@tanstack/react-router";
import { UserLibraryProvider } from "@/contexts/UserLibraryContext";
import { TanStackRouterDevtools } from "@tanstack/react-router-devtools";

import TanstackQueryLayout from "../integrations/tanstack-query/layout";

import type { QueryClient } from "@tanstack/react-query";
import Navbar from "@/components/navbar";

interface MyRouterContext {
	queryClient: QueryClient;
}

export const Route = createRootRouteWithContext<MyRouterContext>()({
	component: () => (
		<UserLibraryProvider>
			<>
				<Navbar />
				<Outlet />
				<TanStackRouterDevtools />

				<TanstackQueryLayout />
			</>
		</UserLibraryProvider>
	),
});
