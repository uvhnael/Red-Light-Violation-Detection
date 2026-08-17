"use client";

import { createContext, useContext, useState } from "react";

interface AISidebarContextValue {
  isOpen: boolean;
  setOpen: (v: boolean) => void;
}

const AISidebarContext = createContext<AISidebarContextValue>({
  isOpen: false,
  setOpen: () => {},
});

export function useAISidebar() {
  return useContext(AISidebarContext);
}

export function AISidebarProvider({ children }: { children: React.ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <AISidebarContext.Provider value={{ isOpen, setOpen: setIsOpen }}>
      {children}
    </AISidebarContext.Provider>
  );
}