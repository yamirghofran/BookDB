import { Outlet, createRootRouteWithContext } from "@tanstack/react-router";
import { TanStackRouterDevtools } from "@tanstack/react-router-devtools";

import TanstackQueryLayout from "../integrations/tanstack-query/layout";

import type { QueryClient } from "@tanstack/react-query";
import Navbar from "@/components/navbar";

interface MyRouterContext {
	queryClient: QueryClient;
}

export const Route = createRootRouteWithContext<MyRouterContext>()({
	component: () => (
		<>
			<Navbar />

			<Outlet />
			<TanStackRouterDevtools />

			<TanstackQueryLayout />
		</>
	),
});
