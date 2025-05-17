import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { fetchPeople, addPerson, type Person, type AddPersonPayload } from "../lib/api";
import { useState } from "react";

export const Route = createFileRoute("/demo/tanstack-query")({
	component: TanStackQueryDemo,
});

function TanStackQueryDemo() {
	const queryClient = useQueryClient();
	const [newName, setNewName] = useState("");
	const [newEmail, setNewEmail] = useState("");

	const { data: people, isLoading, error, isFetching } = useQuery<Person[], Error>({
		queryKey: ["people"],
		queryFn: fetchPeople,
	});

	const addPersonMutation = useMutation<Person, Error, AddPersonPayload>({
		mutationFn: addPerson,
		onSuccess: () => {
			queryClient.invalidateQueries({ queryKey: ["people"] });
			setNewName("");
			setNewEmail("");
		},
		onError: (error) => {
			console.error("Failed to add person:", error.message);
			alert(`Error: ${error.message}`);
		}
	});

	const handleAddPerson = () => {
		const personData: AddPersonPayload = {
			name: newName,
			email: newEmail,
		};
		if (!newName.trim()) {
			alert("Name cannot be empty.");
			return;
		}
		if (!newEmail.trim() || !newEmail.includes('@')) {
			alert("Please enter a valid email.");
			return;
		}
		addPersonMutation.mutate(personData);
	};

	if (isLoading) return <p className="p-4">Loading people...</p>;
	const errorMessage = error?.message || "An unknown error occurred.";
	if (error) return <p className="p-4">Error fetching people: {errorMessage}</p>;

	return (
		<div className="p-4 space-y-4">
			<h1 className="text-2xl mb-4">People List (from Go Backend)</h1>
			{isFetching && <p className="text-sm text-blue-500">Refreshing...</p>}
			<ul className="list-disc pl-5 space-y-1">
				{people?.map((person) => (
					<li key={person.id} className="text-gray-700">
						{person.name} ({person.email})
						{person.age && <span className="text-xs text-gray-500"> - Age: {person.age}</span>}
					</li>
				))}
			</ul>

			<div className="mt-6 p-4 border rounded shadow space-y-3">
				<h2 className="text-xl">Add New Person</h2>
				<div>
					<label htmlFor="name" className="block text-sm font-medium text-gray-700">Name:</label>
					<input 
						type="text" 
						id="name" 
						value={newName} 
						onChange={(e) => setNewName(e.target.value)} 
						className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
						placeholder="Enter name"
					/>
				</div>
				<div>
					<label htmlFor="email" className="block text-sm font-medium text-gray-700">Email:</label>
					<input 
						type="email"
						id="email" 
						value={newEmail} 
						onChange={(e) => setNewEmail(e.target.value)} 
						className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
						placeholder="Enter email"
					/>
				</div>
				<button 
					onClick={handleAddPerson} 
					disabled={addPersonMutation.isPending}
					className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300"
				>
					{addPersonMutation.isPending ? 'Adding...' : 'Add Person'}
				</button>
				{addPersonMutation.isError && (
					<p className="text-red-500">Error: {addPersonMutation.error.message}</p>
				)}
			</div>
		</div>
	);
}
