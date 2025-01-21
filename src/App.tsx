import React, { useState } from 'react';
import Header from './components/Header';
import PostList from './pages/PostList';
import MDFile from './pages/MDFile';
// import './App.css'


const App: React.FC = () => {

  const [selectedFile, setSelectedFile]  = useState<string>('');

  const handleSelectFile = (fileName: string) => {
    setSelectedFile(fileName);
    console.log(`Selected file: ${fileName}`);
  };

  return (
    <div>
      <Header />
      <div className='container'>
        <div className='post-list-container'>
        <PostList onSelect={handleSelectFile} />
        </div>
        <div className='post-container'>
          <MDFile fileName={selectedFile} />
          <div className='main-post'>
            롸그에 오신것을 환영합니다.
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;