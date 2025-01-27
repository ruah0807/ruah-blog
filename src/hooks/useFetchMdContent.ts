import { useEffect, useState } from "react";
import matter from "gray-matter";
import { useParams } from "react-router";

const useFetchMdContent = () => {
  const [content, setContent] = useState<string>("");

  const { category, subtitle } = useParams();
  const fileName = `${category}/${subtitle}`;

  useEffect(() => {
    const fetchContent = async () => {
      if (fileName) {
        const fileModules = import.meta.glob("/posts/**/*.md", {
          query: "?raw",
          import: "default",
        });
        const filePath = `/posts/${fileName}.md`;

        if (fileModules[filePath]) {
          const content = await fileModules[filePath]();
          const { content: mdContent } = matter(content as string);
          setContent(mdContent);
        }
      }
    };
    fetchContent();
  }, [fileName]);

  return { content, fileName };
};

export default useFetchMdContent;
