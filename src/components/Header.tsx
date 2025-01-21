import React from 'react';


const Header: React.FC = () => {
  const handleTitleClick = () => {
    window.location.href = '/'; // Navigate to the main page
  };

  return (
    <header>
      <h1 className='header-title' onClick={handleTitleClick} style={{ cursor: 'pointer' }}>
        롸그</h1>
      <nav>
        <ul>
          <li>Home</li>
          <li>Posts</li>
          <li>About Me</li>
        </ul>
      </nav>
    </header>
  );
};

export default Header; 